"""Unit tests for audio preprocessing and validation."""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from app.medical_audio_annotation.csv_reader import ManifestRow
from app.medical_audio_annotation.preprocessing import AudioPreprocessor


def create_dummy_wav(
    path: Path,
    duration: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    amplitude: float = 0.2,
) -> Path:
    """Generate a clean synthetic sine wave WAV file for testing."""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    data = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    if channels == 2:
        data = np.column_stack([data, data])
    sf.write(str(path), data, sample_rate)
    return path


def test_original_metadata_preserved(tmp_path: Path):
    """Test that original audio properties are recorded and preserved when no transform needed."""
    wav_path = tmp_path / "rec.wav"
    create_dummy_wav(wav_path, duration=1.5, sample_rate=16000, channels=1)

    row = ManifestRow(row_index=1, recording_id="rec", audio_path=wav_path)
    preprocessor = AudioPreprocessor(
        target_sample_rate=None,
        target_channels=None,
        normalize_audio=False,
        trim_silence=False,
        results_dir=tmp_path / "results",
    )

    result = preprocessor.process(row)
    assert result.original_sample_rate == 16000
    assert result.original_channels == 1
    assert result.original_duration_seconds == pytest.approx(1.5, rel=1e-2)
    assert result.processed_duration_seconds == pytest.approx(1.5, rel=1e-2)
    assert result.sample_rate == 16000
    assert not result.is_converted
    assert result.prepared_path == wav_path


def test_normalization_does_not_change_duration(tmp_path: Path):
    """Test that peak normalization scales amplitude without altering audio timeline duration."""
    wav_path = tmp_path / "low_vol.wav"
    create_dummy_wav(wav_path, duration=2.0, sample_rate=16000, amplitude=0.1)

    row = ManifestRow(row_index=1, recording_id="low_vol", audio_path=wav_path)
    preprocessor = AudioPreprocessor(
        normalize_audio=True,
        normalize_target_db=-3.0,
        trim_silence=False,
        results_dir=tmp_path / "results",
    )

    result = preprocessor.process(row)
    assert result.is_converted
    assert result.processed_duration_seconds == pytest.approx(result.original_duration_seconds, rel=1e-3)

    # Verify peak amplitude matches -3 dBFS (approx 0.7079)
    norm_data, _ = sf.read(str(result.prepared_path))
    peak = np.max(np.abs(norm_data))
    expected_peak = 10.0 ** (-3.0 / 20.0)
    assert peak == pytest.approx(expected_peak, rel=1e-2)


def test_trim_silence_raises_error(tmp_path: Path):
    """Test that enabling TRIM_SILENCE raises ValueError since restoration is unsupported."""
    wav_path = tmp_path / "rec.wav"
    create_dummy_wav(wav_path, duration=1.0)

    row = ManifestRow(row_index=1, recording_id="rec", audio_path=wav_path)
    preprocessor = AudioPreprocessor(trim_silence=True, results_dir=tmp_path)

    with pytest.raises(ValueError, match="TRIM_SILENCE is enabled.*no timestamp restoration"):
        preprocessor.process(row)


def test_file_size_validation(tmp_path: Path):
    """Test rejection of audio files exceeding max_audio_bytes."""
    wav_path = tmp_path / "large.wav"
    create_dummy_wav(wav_path, duration=1.0)

    row = ManifestRow(row_index=1, recording_id="large", audio_path=wav_path)
    # Set threshold lower than file size
    preprocessor = AudioPreprocessor(max_audio_bytes=100, results_dir=tmp_path)

    with pytest.raises(ValueError, match="exceeds maximum limit"):
        preprocessor.process(row)


def test_unsupported_extension(tmp_path: Path):
    """Test rejection of unsupported audio extensions."""
    txt_path = tmp_path / "not_audio.txt"
    txt_path.write_text("not audio content", encoding="utf-8")

    row = ManifestRow(row_index=1, recording_id="invalid_ext", audio_path=txt_path)
    preprocessor = AudioPreprocessor(results_dir=tmp_path)

    with pytest.raises(ValueError, match="Unsupported audio format/extension"):
        preprocessor.process(row)
