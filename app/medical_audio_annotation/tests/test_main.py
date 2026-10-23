"""End-to-end integration tests for the annotation pipeline."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from app.medical_audio_annotation.inference import AnnotationResponse
from app.medical_audio_annotation.main import PipelineRunner


def create_audio_file(path: Path, duration: float = 1.0, sample_rate: int = 16000) -> None:
    """Helper to generate a real WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(duration * sample_rate)
    data = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples, endpoint=False))).astype(
        np.float32
    )
    sf.write(str(path), data, sample_rate)


def test_pipeline_runner_end_to_end_mocked(tmp_path: Path):
    """Test full pipeline execution verifying output JSONs written for each manifest row."""
    upload_dir = tmp_path / "upload"
    results_dir = tmp_path / "results"
    csv_path = upload_dir / "audio_manifest.csv"

    # Create dummy audio files
    create_audio_file(upload_dir / "rec1.wav", duration=1.0)
    create_audio_file(upload_dir / "rec2.wav", duration=2.0)

    csv_path.write_text(
        "audio_path,recording_id\n"
        "rec1.wav,rec1\n"
        "rec2.wav,rec2\n",
        encoding="utf-8",
    )

    mock_events_1 = [
        {
            "event_id": "e_01",
            "label": "provisional_wheeze",
            "description": "Continuous musical tone",
            "start_seconds": 0.2,
            "end_seconds": 0.8,
            "uncertain": False,
            "uncertainty_reason": None,
            "confidence_score": 0.9,
        }
    ]

    mock_events_2 = [
        {
            "event_id": "e_02",
            "label": "provisional_crackles",
            "description": "Discontinuous explosive sound",
            "start_seconds": 0.5,
            "end_seconds": 1.5,
            "uncertain": True,
            "uncertainty_reason": "Low signal-to-noise ratio",
            "confidence_score": 0.5,
        }
    ]

    def mock_annotate(self, result, system_prompt, user_message):
        events = mock_events_1 if result.recording_id == "rec1" else mock_events_2
        return AnnotationResponse(
            recording_id=result.recording_id,
            raw_json=json.dumps({"recording_id": result.recording_id, "events": events}),
            events=events,
            model="gpt-4o-audio-preview",
            usage={"total_tokens": 100},
        )

    with patch(
        "app.medical_audio_annotation.main.OpenAIAudioAnnotator.annotate",
        side_effect=mock_annotate,
        autospec=True,
    ), patch(
        "app.medical_audio_annotation.config.OPENAI_API_KEY",
        "sk-mock-key-for-test-1234567890",
    ):
        runner = PipelineRunner(
            csv_path=csv_path,
            upload_dir=upload_dir,
            results_dir=results_dir,
            dry_run=False,
            overwrite=False,
        )
        summary = runner.run()

    assert summary["manifest_rows"] == 2
    assert summary["selected_rows"] == 2
    assert summary["succeeded"] == 2
    assert summary["skipped"] == 0
    assert summary["failed"] == 0

    # Verify output JSON files
    res1_path = results_dir / "rec1.json"
    res2_path = results_dir / "rec2.json"
    assert res1_path.is_file()
    assert res2_path.is_file()

    with open(res1_path, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    assert data1["recording_id"] == "rec1"
    assert data1["status"] == "success"
    assert len(data1["events"]) == 1
    assert data1["events"][0]["label"] == "provisional_wheeze"
    assert data1["original_duration_seconds"] == pytest.approx(1.0, rel=1e-2)

    with open(res2_path, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    assert data2["recording_id"] == "rec2"
    assert data2["status"] == "success"
    assert len(data2["events"]) == 1
    assert data2["events"][0]["label"] == "provisional_crackles"
    assert data2["events"][0]["uncertain"] is True


def test_pipeline_runner_skips_existing_results(tmp_path: Path):
    """Test that existing valid results are skipped on re-run when overwrite=False."""
    upload_dir = tmp_path / "upload"
    results_dir = tmp_path / "results"
    csv_path = upload_dir / "audio_manifest.csv"

    create_audio_file(upload_dir / "rec1.wav", duration=1.0)
    csv_path.write_text("audio_path,recording_id\nrec1.wav,rec1\n", encoding="utf-8")

    # First run in dry-run mode
    runner = PipelineRunner(
        csv_path=csv_path,
        upload_dir=upload_dir,
        results_dir=results_dir,
        dry_run=True,
        overwrite=False,
    )
    summary1 = runner.run()
    assert summary1["succeeded"] == 1
    assert summary1["skipped"] == 0

    # Second run should skip
    summary2 = runner.run()
    assert summary2["succeeded"] == 0
    assert summary2["skipped"] == 1
