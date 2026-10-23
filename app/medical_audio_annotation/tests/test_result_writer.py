"""Unit tests for result serialization, validation, and collision avoidance."""

import json
import pytest
from pathlib import Path

from app.medical_audio_annotation.csv_reader import ManifestRow
from app.medical_audio_annotation.result_writer import AnnotationResult, ResultWriter


def create_sample_result(
    recording_id: str = "rec_001",
    status: str = "success",
    duration: float = 5.0,
    events: list | None = None,
) -> AnnotationResult:
    """Helper to build a valid AnnotationResult."""
    if events is None:
        events = [
            {
                "event_id": "evt_1",
                "label": "provisional_wheeze",
                "description": "High pitched continuous sound",
                "start_seconds": 1.0,
                "end_seconds": 2.5,
                "uncertain": False,
            }
        ]
    return AnnotationResult(
        schema_version="1.0",
        recording_id=recording_id,
        source_file=f"upload/{recording_id}.wav",
        status=status,
        model="gpt-4o-audio-preview",
        preprocessing_version="1.0",
        original_duration_seconds=duration,
        events=events,
        error=None,
    )


def test_atomic_write(tmp_path: Path):
    """Test that atomic write creates valid JSON and cleans up any temp files."""
    results_dir = tmp_path / "results"
    writer = ResultWriter(results_dir=results_dir)

    row = ManifestRow(row_index=1, recording_id="rec_001", audio_path=tmp_path / "rec_001.wav")
    res = create_sample_result(recording_id="rec_001")

    written_path = writer.write(res, row)
    assert written_path.is_file()
    assert written_path == results_dir / "rec_001.json"

    # Verify no dangling temp files
    tmp_files = list(results_dir.glob("*.tmp"))
    assert len(tmp_files) == 0

    with open(written_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["recording_id"] == "rec_001"
    assert data["status"] == "success"
    assert len(data["events"]) == 1


def test_is_valid_result(tmp_path: Path):
    """Test validity checking logic on results."""
    results_dir = tmp_path / "results"
    writer = ResultWriter(results_dir=results_dir)

    # 1. Non-existent path
    assert not writer.is_valid_result(results_dir / "missing.json")

    # 2. Corrupt JSON
    bad_json = results_dir / "bad.json"
    bad_json.parent.mkdir(parents=True, exist_ok=True)
    bad_json.write_text("{corrupt json", encoding="utf-8")
    assert not writer.is_valid_result(bad_json)

    # 3. Missing required fields
    incomplete = results_dir / "incomplete.json"
    incomplete.write_text(json.dumps({"recording_id": "rec_001"}), encoding="utf-8")
    assert not writer.is_valid_result(incomplete)

    # 4. Status == "error" is NOT a valid result (must be retried)
    error_file = results_dir / "error.json"
    error_payload = {
        "schema_version": "1.0",
        "recording_id": "rec_001",
        "source_file": "file.wav",
        "status": "error",
        "model": "model",
        "preprocessing_version": "1.0",
        "original_duration_seconds": 1.0,
        "events": [],
        "error": {"code": "ERR", "message": "msg"},
    }
    error_file.write_text(json.dumps(error_payload), encoding="utf-8")
    assert not writer.is_valid_result(error_file)

    # 5. Full valid result
    valid_file = results_dir / "valid.json"
    valid_payload = dict(error_payload)
    valid_payload["status"] = "success"
    valid_file.write_text(json.dumps(valid_payload), encoding="utf-8")
    assert writer.is_valid_result(valid_file)


def test_error_result_writing_and_sanitization(tmp_path: Path):
    """Test error writing with API key redaction."""
    results_dir = tmp_path / "results"
    writer = ResultWriter(results_dir=results_dir)
    row = ManifestRow(row_index=1, recording_id="rec_err", audio_path=tmp_path / "rec_err.wav")

    leaked_msg = "Failed with key sk-abcdef1234567890abcdef123456 during inference"
    error_path = writer.write_error(row, code="AUTH_FAILED", message=leaked_msg)

    assert error_path.is_file()
    with open(error_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "error"
    assert data["error"]["code"] == "AUTH_FAILED"
    assert "sk-abcdef1234567890abcdef123456" not in data["error"]["message"]
    assert "[REDACTED_API_KEY]" in data["error"]["message"]


def test_collision_avoidance(tmp_path: Path):
    """Test that identical stems in different subfolders generate distinct paths."""
    upload_dir = tmp_path / "upload"
    results_dir = tmp_path / "results"
    writer = ResultWriter(results_dir=results_dir, upload_dir=upload_dir)

    row1 = ManifestRow(
        row_index=1,
        recording_id="sample",
        audio_path=upload_dir / "site_A" / "sample.wav",
    )
    row2 = ManifestRow(
        row_index=2,
        recording_id="sample",
        audio_path=upload_dir / "site_B" / "sample.wav",
    )

    path1 = writer.get_result_path(row1)
    path2 = writer.get_result_path(row2)

    assert path1 != path2
    assert "site_A" in str(path1)
    assert "site_B" in str(path2)


def test_timestamp_validation(tmp_path: Path):
    """Test timestamp boundary validation."""
    results_dir = tmp_path / "results"
    writer = ResultWriter(results_dir=results_dir)
    row = ManifestRow(row_index=1, recording_id="rec_time", audio_path=tmp_path / "rec_time.wav")

    # 1. Negative start_seconds
    bad_res1 = create_sample_result(
        recording_id="rec_time",
        duration=5.0,
        events=[{
            "event_id": "e1",
            "label": "sound",
            "description": "desc",
            "start_seconds": -0.5,
            "end_seconds": 2.0,
            "uncertain": False,
        }],
    )
    p1 = writer.write(bad_res1, row)
    with open(p1, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "error"
    assert data["error"]["code"] == "VALIDATION_ERROR"

    # 2. end_seconds <= start_seconds
    bad_res2 = create_sample_result(
        recording_id="rec_time",
        duration=5.0,
        events=[{
            "event_id": "e1",
            "label": "sound",
            "description": "desc",
            "start_seconds": 3.0,
            "end_seconds": 2.0,
            "uncertain": False,
        }],
    )
    p2 = writer.write(bad_res2, row)
    with open(p2, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "error"

    # 3. end_seconds exceeds duration + 0.1s tolerance
    bad_res3 = create_sample_result(
        recording_id="rec_time",
        duration=5.0,
        events=[{
            "event_id": "e1",
            "label": "sound",
            "description": "desc",
            "start_seconds": 1.0,
            "end_seconds": 5.2,
            "uncertain": False,
        }],
    )
    p3 = writer.write(bad_res3, row)
    with open(p3, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "error"

    # 4. Within tolerance (duration=5.0, end_seconds=5.08)
    good_res = create_sample_result(
        recording_id="rec_time",
        duration=5.0,
        events=[{
            "event_id": "e1",
            "label": "sound",
            "description": "desc",
            "start_seconds": 1.0,
            "end_seconds": 5.08,
            "uncertain": False,
        }],
    )
    p4 = writer.write(good_res, row)
    with open(p4, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "success"
