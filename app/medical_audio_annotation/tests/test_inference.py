"""Unit tests for inference client, retry logic, and error handling with mocks."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import BadRequestError, RateLimitError

from app.medical_audio_annotation.inference import OpenAIAudioAnnotator
from app.medical_audio_annotation.preprocessing import PreprocessingResult


@pytest.fixture
def dummy_prep_result(tmp_path: Path) -> PreprocessingResult:
    """Fixture providing a dummy PreprocessingResult with an audio file on disk."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"dummy wav bytes")
    return PreprocessingResult(
        recording_id="test_rec",
        source_path=audio_file,
        prepared_path=audio_file,
        original_sample_rate=16000,
        original_channels=1,
        original_duration_seconds=2.0,
        processed_duration_seconds=2.0,
        sample_rate=16000,
        is_converted=False,
        preprocessing_version="1.0",
    )


def test_dry_run_returns_mock(dummy_prep_result: PreprocessingResult):
    """Test that DRY_RUN mode immediately returns a mock AnnotationResponse."""
    annotator = OpenAIAudioAnnotator(api_key="dummy_key", dry_run=True)
    response = annotator.annotate(
        result=dummy_prep_result,
        system_prompt="sys prompt",
        user_message="user prompt",
    )
    assert response.recording_id == "test_rec"
    assert response.events == []
    assert "mock" in response.model
    assert response.usage is not None
    assert response.usage["total_tokens"] == 0


def test_retry_on_rate_limit_error(dummy_prep_result: PreprocessingResult):
    """Test that transient RateLimitError triggers bounded retries and succeeds."""
    annotator = OpenAIAudioAnnotator(
        api_key="fake_key",
        timeout=10,
        max_retries=3,
        backoff=0.01,
        dry_run=False,
    )

    mock_req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_resp = httpx.Response(status_code=429, request=mock_req)
    rate_err = RateLimitError("Rate limit exceeded", response=mock_resp, body=None)

    success_choice = MagicMock()
    success_choice.message.content = json.dumps({
        "recording_id": "test_rec",
        "events": [],
    })
    success_response = MagicMock()
    success_response.choices = [success_choice]
    success_response.model = "gpt-4o-audio-preview"
    success_response.usage = MagicMock(model_dump=lambda: {"total_tokens": 50})

    mock_client = MagicMock()
    # Fails twice with RateLimitError, then succeeds
    mock_client.chat.completions.create.side_effect = [
        rate_err,
        rate_err,
        success_response,
    ]
    annotator.client = mock_client

    with patch("time.sleep", return_value=None):
        result = annotator.annotate(dummy_prep_result, "sys", "usr")

    assert result.recording_id == "test_rec"
    assert mock_client.chat.completions.create.call_count == 3


def test_no_retry_on_bad_request_error(dummy_prep_result: PreprocessingResult):
    """Test that non-retryable BadRequestError raises immediately without retrying."""
    annotator = OpenAIAudioAnnotator(
        api_key="fake_key",
        timeout=10,
        max_retries=3,
        backoff=0.01,
        dry_run=False,
    )

    mock_req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_resp = httpx.Response(status_code=400, request=mock_req)
    bad_req_err = BadRequestError("Invalid format", response=mock_resp, body=None)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = bad_req_err
    annotator.client = mock_client

    with pytest.raises(BadRequestError):
        annotator.annotate(dummy_prep_result, "sys", "usr")

    # Should have executed only 1 attempt
    assert mock_client.chat.completions.create.call_count == 1


def test_json_parsing_and_markdown_stripping(dummy_prep_result: PreprocessingResult):
    """Test parsing JSON with markdown fences."""
    annotator = OpenAIAudioAnnotator(api_key="fake_key", dry_run=False)

    fenced_content = """```json
{
  "recording_id": "test_rec",
  "events": [
    {
      "event_id": "e_01",
      "label": "provisional_crackles",
      "description": "Intermittent explosive click sound",
      "start_seconds": 0.2,
      "end_seconds": 0.8,
      "uncertain": false,
      "uncertainty_reason": null,
      "confidence_score": 0.85
    }
  ]
}
```"""
    success_choice = MagicMock()
    success_choice.message.content = fenced_content
    success_response = MagicMock()
    success_response.choices = [success_choice]
    success_response.model = "gpt-4o-audio-preview"
    success_response.usage = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = success_response
    annotator.client = mock_client

    result = annotator.annotate(dummy_prep_result, "sys", "usr")
    assert len(result.events) == 1
    assert result.events[0]["label"] == "provisional_crackles"
    assert result.events[0]["start_seconds"] == 0.2


def test_usage_tracking(dummy_prep_result: PreprocessingResult):
    """Test extraction of token usage metrics."""
    annotator = OpenAIAudioAnnotator(api_key="fake_key", dry_run=False)

    success_choice = MagicMock()
    success_choice.message.content = json.dumps({"recording_id": "test_rec", "events": []})
    success_response = MagicMock()
    success_response.choices = [success_choice]
    success_response.model = "gpt-4o-audio-preview"
    success_response.usage = MagicMock(
        model_dump=lambda: {
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
        }
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = success_response
    annotator.client = mock_client

    result = annotator.annotate(dummy_prep_result, "sys", "usr")
    assert result.usage == {
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "total_tokens": 165,
    }
