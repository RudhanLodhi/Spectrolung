"""Unit tests for configuration loading and validation."""

import pytest
from pathlib import Path

from app.medical_audio_annotation import config
from app.medical_audio_annotation.config import (
    compute_effective_end_index,
    count_csv_data_rows,
    validate_config,
)


def test_count_csv_data_rows(tmp_path: Path):
    """Test data row counting with header and non-empty lines."""
    csv_file = tmp_path / "manifest.csv"
    csv_file.write_text(
        "audio_path,recording_id\n"
        "sample1.wav,rec1\n"
        "sample2.wav,rec2\n"
        "sample3.wav,rec3\n",
        encoding="utf-8",
    )
    assert count_csv_data_rows(csv_file) == 3

    # CSV with header only
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("audio_path,recording_id\n", encoding="utf-8")
    assert count_csv_data_rows(header_only) == 0

    # Empty file
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert count_csv_data_rows(empty) == 0


def test_missing_csv_handling(tmp_path: Path):
    """Test that missing CSV returns 0 and issues a warning without raising an exception."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.warns(UserWarning, match="CSV manifest file not found"):
        count = count_csv_data_rows(non_existent)
    assert count == 0


def test_effective_end_index_computation():
    """Test EFFECTIVE_END_INDEX calculation under various boundary conditions."""
    # None should default to all_count
    assert compute_effective_end_index(None, 10) == 10
    assert compute_effective_end_index(None, 0) == 0

    # Within bounds
    assert compute_effective_end_index(5, 10) == 5

    # Exceeding all_count clamps to all_count
    assert compute_effective_end_index(20, 10) == 10
    assert compute_effective_end_index(5, 0) == 0


def test_validate_config_success():
    """Test successful configuration validation."""
    effective = validate_config(
        start_index=0,
        end_index=5,
        all_count=10,
        dry_run=True,
        api_key="",
        trim_silence=False,
    )
    assert effective == 5


def test_validate_config_errors():
    """Test validation errors for invalid indices and required credentials."""
    # Negative start index
    with pytest.raises(ValueError, match="START_INDEX.*cannot be negative"):
        validate_config(start_index=-1, end_index=5, all_count=10, dry_run=True)

    # Negative end index
    with pytest.raises(ValueError, match="END_INDEX.*cannot be negative"):
        validate_config(start_index=0, end_index=-1, all_count=10, dry_run=True)

    # Start index exceeds effective end index
    with pytest.raises(ValueError, match="START_INDEX.*exceeds EFFECTIVE_END_INDEX"):
        validate_config(start_index=6, end_index=4, all_count=10, dry_run=True)

    # Missing API key when dry_run is False
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        validate_config(
            start_index=0,
            end_index=2,
            all_count=5,
            dry_run=False,
            api_key="",
        )

    # Silence trimming enabled raises error
    with pytest.raises(ValueError, match="TRIM_SILENCE is enabled"):
        validate_config(
            start_index=0,
            end_index=2,
            all_count=5,
            dry_run=True,
            trim_silence=True,
        )
