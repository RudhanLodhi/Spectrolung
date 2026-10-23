"""Result serialization, atomic writing, and validation for medical audio annotations."""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from . import config
from .csv_reader import ManifestRow

logger = logging.getLogger(__name__)

API_KEY_REGEX = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")


@dataclass
class AnnotationResult:
    """Represents the complete result of an annotation attempt."""

    schema_version: str
    recording_id: str
    source_file: str
    status: str  # "success" | "error" | "skipped" | "dry_run"
    model: str
    preprocessing_version: str
    original_duration_seconds: Optional[float]
    events: list[dict[str, Any]]
    error: Optional[dict[str, str]]  # {"code": str, "message": str} or None


class ResultWriter:
    """Handles writing, validation, and collision-avoidance for annotation results."""

    REQUIRED_FIELDS: list[str] = [
        "schema_version",
        "recording_id",
        "source_file",
        "status",
        "model",
        "preprocessing_version",
        "original_duration_seconds",
        "events",
        "error",
    ]

    def __init__(
        self,
        results_dir: Path,
        overwrite: bool = False,
        upload_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the ResultWriter.

        Args:
            results_dir: Directory where results will be stored.
            overwrite: Whether existing valid results should be overwritten.
            upload_dir: Upload directory for relative path calculation.
        """
        self.results_dir = Path(results_dir)
        self.overwrite = overwrite
        self.upload_dir = Path(upload_dir) if upload_dir else config.UPLOAD_DIR

    def sanitize_message(self, message: str) -> str:
        """Strip API keys or sensitive credential patterns from error messages."""
        if not message:
            return ""
        sanitized = API_KEY_REGEX.sub("[REDACTED_API_KEY]", str(message))
        if config.OPENAI_API_KEY and len(config.OPENAI_API_KEY) > 5:
            sanitized = sanitized.replace(config.OPENAI_API_KEY, "[REDACTED_API_KEY]")
        return sanitized

    def get_result_path(self, row: ManifestRow) -> Path:
        """Determine unique output result path for a manifest row.

        If files share the same recording_id/stem but originate from distinct
        subfolders, the parent subfolder path is mirrored under results_dir to avoid collision.

        Args:
            row: ManifestRow instance.

        Returns:
            Path where the result JSON should be saved.
        """
        # Case 1: row.audio_path is within upload_dir
        try:
            rel = row.audio_path.resolve().relative_to(self.upload_dir.resolve())
            if rel.parent != Path(".") and str(rel.parent) not in ("", "."):
                return self.results_dir / rel.parent / f"{row.recording_id}.json"
        except (ValueError, AttributeError):
            pass

        # Case 2: relative audio path with parent directories
        if (
            not row.audio_path.is_absolute()
            and row.audio_path.parent != Path(".")
            and str(row.audio_path.parent) not in ("", ".")
        ):
            return self.results_dir / row.audio_path.parent / f"{row.recording_id}.json"

        return self.results_dir / f"{row.recording_id}.json"

    def result_exists(self, row: ManifestRow) -> bool:
        """Check if a result file already exists on disk for the given row."""
        return self.get_result_path(row).is_file()

    def is_valid_result(self, path: Path) -> bool:
        """Determine if a file is a valid previously completed result.

        Error results or malformed files are considered invalid so they can be retried.

        Args:
            path: Path to check.

        Returns:
            True if valid, False otherwise.
        """
        if not path.is_file():
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return False

        if not isinstance(data, dict):
            return False

        for field_name in self.REQUIRED_FIELDS:
            if field_name not in data:
                return False

        if data.get("status") == "error":
            return False

        if not isinstance(data.get("events"), list):
            return False

        return True

    def validate_result(self, result: AnnotationResult) -> tuple[bool, Optional[str]]:
        """Validate structure and timestamp constraints of an AnnotationResult.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if result.status == "error":
            return True, None

        if not isinstance(result.events, list):
            return False, "events field must be a list"

        duration = result.original_duration_seconds
        tolerance = 0.1

        for idx, event in enumerate(result.events):
            if not isinstance(event, dict):
                return False, f"Event {idx} is not a dictionary"

            for req in ("event_id", "label", "description", "start_seconds", "end_seconds", "uncertain"):
                if req not in event:
                    return False, f"Event {idx} missing required key '{req}'"

            if not isinstance(event["event_id"], str) or not event["event_id"].strip():
                return False, f"Event {idx}: event_id must be non-empty string"

            if not isinstance(event["label"], str) or not event["label"].strip():
                return False, f"Event {idx}: label must be non-empty string"

            if not isinstance(event["description"], str) or not event["description"].strip():
                return False, f"Event {idx}: description must be non-empty string"

            if not isinstance(event["uncertain"], bool):
                return False, f"Event {idx}: uncertain must be a boolean"

            try:
                start = float(event["start_seconds"])
                end = float(event["end_seconds"])
            except (ValueError, TypeError):
                return False, f"Event {idx}: start_seconds and end_seconds must be numbers"

            if start < 0.0:
                return False, f"Event {idx}: start_seconds ({start}) cannot be negative"

            if end <= start:
                return False, f"Event {idx}: end_seconds ({end}) must be greater than start_seconds ({start})"

            if duration is not None:
                max_allowed = duration + tolerance
                if start > max_allowed:
                    return False, f"Event {idx}: start_seconds ({start}) exceeds audio duration ({duration})"
                if end > max_allowed:
                    return False, f"Event {idx}: end_seconds ({end}) exceeds audio duration ({duration})"

        return True, None

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        """Write dictionary payload as JSON atomically using a temporary file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, path)
            return path
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def write(self, result: AnnotationResult, row: ManifestRow) -> Path:
        """Validate and write annotation result atomically.

        If validation fails, writes an error result instead.

        Args:
            result: AnnotationResult to persist.
            row: Corresponding ManifestRow.

        Returns:
            Path of the written result file.
        """
        is_valid, error_msg = self.validate_result(result)
        if not is_valid:
            logger.warning(
                "Result validation failed for %s: %s. Writing error result.",
                row.recording_id,
                error_msg,
            )
            return self.write_error(
                row=row,
                code="VALIDATION_ERROR",
                message=error_msg or "Validation failed",
                schema_version=result.schema_version,
                model=result.model,
                preprocessing_version=result.preprocessing_version,
                original_duration_seconds=result.original_duration_seconds,
            )

        target_path = self.get_result_path(row)
        payload = asdict(result)
        return self._atomic_write_json(target_path, payload)

    def write_error(
        self,
        row: ManifestRow,
        code: str,
        message: str,
        schema_version: str = "1.0",
        model: str = "unknown",
        preprocessing_version: str = config.PREPROCESSING_VERSION,
        original_duration_seconds: Optional[float] = None,
    ) -> Path:
        """Write an error result for a recording.

        Args:
            row: ManifestRow for the recording.
            code: Short error category code.
            message: Descriptive error message.
            schema_version: Schema version string.
            model: Model name.
            preprocessing_version: Version of preprocessing applied.
            original_duration_seconds: Duration if known.

        Returns:
            Path of written error file.
        """
        sanitized_msg = self.sanitize_message(message)
        error_result = AnnotationResult(
            schema_version=schema_version,
            recording_id=row.recording_id,
            source_file=str(row.audio_path),
            status="error",
            model=model,
            preprocessing_version=preprocessing_version,
            original_duration_seconds=original_duration_seconds,
            events=[],
            error={"code": code, "message": sanitized_msg},
        )
        target_path = self.get_result_path(row)
        return self._atomic_write_json(target_path, asdict(error_result))
