"""LLM audio inference client for medical audio annotation pipeline."""

import base64
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from .preprocessing import PreprocessingResult

logger = logging.getLogger(__name__)


@dataclass
class AnnotationResponse:
    """Encapsulates raw and parsed response from the audio LLM."""

    recording_id: str
    raw_json: str
    events: list[dict]
    model: str
    usage: Optional[dict[str, Any]]


class OpenAIAudioAnnotator:
    """Interacts with OpenAI audio models with retry and error handling."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-audio-preview",
        timeout: int = 120,
        max_retries: int = 3,
        backoff: float = 2.0,
        dry_run: bool = False,
    ) -> None:
        """Initialize the OpenAI audio annotator.

        Args:
            api_key: OpenAI API key.
            model: Model identifier.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for transient errors.
            backoff: Exponential backoff base factor in seconds.
            dry_run: If True, returns mock responses without network calls.
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.dry_run = dry_run
        self.client: Optional[OpenAI] = (
            None if dry_run else OpenAI(api_key=api_key, timeout=float(timeout))
        )

    def annotate(
        self,
        result: PreprocessingResult,
        system_prompt: str,
        user_message: str,
    ) -> AnnotationResponse:
        """Annotate preprocessed audio using OpenAI Chat Completions API.

        Args:
            result: PreprocessingResult containing audio file and metadata.
            system_prompt: Full system instructions prompt.
            user_message: Per-recording user prompt text.

        Returns:
            AnnotationResponse with parsed events and token usage.

        Raises:
            AuthenticationError: On invalid API credentials (non-retryable).
            BadRequestError: On malformed requests (non-retryable).
            RateLimitError: If rate limit retries are exhausted.
            ValueError: If the response cannot be parsed as JSON.
        """
        if self.dry_run:
            logger.info("DRY_RUN enabled: returning mock annotation response for %s", result.recording_id)
            mock_data = {
                "recording_id": result.recording_id,
                "events": [],
            }
            return AnnotationResponse(
                recording_id=result.recording_id,
                raw_json=json.dumps(mock_data),
                events=[],
                model=f"{self.model}-mock",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        # Read and encode audio file to base64
        audio_bytes = result.prepared_path.read_bytes()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        fmt = result.prepared_path.suffix.lstrip(".").lower()
        if fmt not in ("wav", "mp3", "flac", "ogg", "m4a", "webm"):
            fmt = "wav"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_audio,
                            "format": fmt,
                        },
                    },
                ],
            },
        ]

        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                assert self.client is not None
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    response_format={"type": "json_object"},
                    timeout=float(self.timeout),
                )
                break
            except (AuthenticationError, BadRequestError) as exc:
                logger.error(
                    "Permanent non-retryable error during inference for %s: %s",
                    result.recording_id,
                    exc,
                )
                raise
            except (
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                InternalServerError,
            ) as exc:
                last_exception = exc
                if attempt >= self.max_retries:
                    logger.error(
                        "Max retries (%d) exhausted for %s: %s",
                        self.max_retries,
                        result.recording_id,
                        exc,
                    )
                    raise
                sleep_sec = self.backoff * (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "Retryable error %s on attempt %d/%d for %s. Retrying in %.2fs...",
                    type(exc).__name__,
                    attempt + 1,
                    self.max_retries,
                    result.recording_id,
                    sleep_sec,
                )
                time.sleep(sleep_sec)

        content = response.choices[0].message.content or ""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse model response as JSON: {exc}") from exc

        events = parsed.get("events", [])
        if not isinstance(events, list):
            raise ValueError(f"Model output 'events' field must be a list, got {type(events).__name__}")

        usage_dict = None
        if getattr(response, "usage", None):
            if hasattr(response.usage, "model_dump"):
                usage_dict = response.usage.model_dump()
            elif isinstance(response.usage, dict):
                usage_dict = response.usage
            else:
                try:
                    usage_dict = dict(response.usage)
                except Exception:
                    usage_dict = {
                        "total_tokens": getattr(response.usage, "total_tokens", None)
                    }

        return AnnotationResponse(
            recording_id=result.recording_id,
            raw_json=cleaned,
            events=events,
            model=response.model or self.model,
            usage=usage_dict,
        )
