"""Unit tests for CSV manifest reading and validation."""

import pytest
from pathlib import Path

from app.medical_audio_annotation.csv_reader import ManifestReader, ManifestRow


def test_normal_load(tmp_path: Path):
    """Test loading valid manifest with existing audio files."""
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    (upload_dir / "rec1.wav").write_bytes(b"dummy audio 1")
    (upload_dir / "rec2.wav").write_bytes(b"dummy audio 2")

    csv_path = tmp_path / "manifest.csv"
    csv_path.write_text(
        "audio_path,recording_id\n"
        "rec1.wav,id_001\n"
        "rec2.wav,id_002\n",
        encoding="utf-8",
    )

    reader = ManifestReader(
        csv_path=csv_path,
        upload_dir=upload_dir,
        audio_path_col="audio_path",
        id_col="recording_id",
        allow_absolute=False,
    )

    rows = reader.load()
    assert len(rows) == 2
    assert rows[0].recording_id == "id_001"
    assert rows[0].audio_path == (upload_dir / "rec1.wav").resolve()
    assert rows[0].row_index == 1
    assert rows[1].recording_id == "id_002"
    assert rows[1].audio_path == (upload_dir / "rec2.wav").resolve()


def test_missing_column(tmp_path: Path):
    """Test error when required audio_path column is missing."""
    csv_path = tmp_path / "bad_manifest.csv"
    csv_path.write_text("file_name,recording_id\nrec1.wav,id_001\n", encoding="utf-8")

    reader = ManifestReader(
        csv_path=csv_path,
        upload_dir=tmp_path,
        audio_path_col="audio_path",
        id_col="recording_id",
        allow_absolute=False,
    )

    with pytest.raises(ValueError, match="Missing required audio path column"):
        reader.load()


def test_empty_path(tmp_path: Path):
    """Test error when audio path cell is empty."""
    csv_path = tmp_path / "empty_path.csv"
    csv_path.write_text("audio_path,recording_id\n ,id_001\n", encoding="utf-8")

    reader = ManifestReader(
        csv_path=csv_path,
        upload_dir=tmp_path,
        audio_path_col="audio_path",
        id_col="recording_id",
        allow_absolute=False,
    )

    with pytest.raises(ValueError, match="Empty audio path"):
        reader.load()


def test_duplicate_ids(tmp_path: Path):
    """Test error when duplicate recording IDs are encountered."""
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    (upload_dir / "rec1.wav").write_bytes(b"dummy audio 1")
    (upload_dir / "rec2.wav").write_bytes(b"dummy audio 2")

    csv_path = tmp_path / "dup_manifest.csv"
    csv_path.write_text(
        "audio_path,recording_id\n"
        "rec1.wav,id_dup\n"
        "rec2.wav,id_dup\n",
        encoding="utf-8",
    )

    reader = ManifestReader(
        csv_path=csv_path,
        upload_dir=upload_dir,
        audio_path_col="audio_path",
        id_col="recording_id",
        allow_absolute=False,
    )

    with pytest.raises(ValueError, match="Duplicate recording_id encountered"):
        reader.load()


def test_path_traversal(tmp_path: Path):
    """Test rejection of path traversal sequences trying to escape upload_dir."""
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()

    csv_path = tmp_path / "traversal.csv"
    csv_path.write_text("audio_path,recording_id\n../outside.wav,id_001\n", encoding="utf-8")

    reader = ManifestReader(
        csv_path=csv_path,
        upload_dir=upload_dir,
        audio_path_col="audio_path",
        id_col="recording_id",
        allow_absolute=False,
    )

    with pytest.raises(ValueError, match="Path traversal detected"):
        reader.load()


def test_absolute_paths_when_disallowed(tmp_path: Path):
    """Test that absolute paths raise ValueError when allow_absolute is False."""
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    sample_file = tmp_path / "outside.wav"
    sample_file.write_bytes(b"dummy")

    csv_path = tmp_path / "abs_manifest.csv"
    # Format absolute path
    abs_str = str(sample_file.resolve()).replace("\\", "/")
    csv_path.write_text(f"audio_path,recording_id\n{abs_str},id_001\n", encoding="utf-8")

    reader = ManifestReader(
        csv_path=csv_path,
        upload_dir=upload_dir,
        audio_path_col="audio_path",
        id_col="recording_id",
        allow_absolute=False,
    )

    with pytest.raises(ValueError, match="Absolute audio path not allowed"):
        reader.load()


def test_extra_columns(tmp_path: Path):
    """Test extraction of additional non-standard metadata columns."""
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    (upload_dir / "rec1.wav").write_bytes(b"dummy")

    csv_path = tmp_path / "extra.csv"
    csv_path.write_text(
        "audio_path,recording_id,patient_age,site\n"
        "rec1.wav,id_001,65,trachea\n",
        encoding="utf-8",
    )

    reader = ManifestReader(
        csv_path=csv_path,
        upload_dir=upload_dir,
        audio_path_col="audio_path",
        id_col="recording_id",
        allow_absolute=False,
    )

    rows = reader.load()
    assert len(rows) == 1
    assert rows[0].extra_fields == {"patient_age": "65", "site": "trachea"}


def test_subfolder_paths_and_derived_id(tmp_path: Path):
    """Test resolution of nested audio files and ID derivation when id column is blank."""
    upload_dir = tmp_path / "upload"
    sub_dir = upload_dir / "subfolder"
    sub_dir.mkdir(parents=True)
    (sub_dir / "nested_audio.wav").write_bytes(b"dummy")

    csv_path = tmp_path / "subfolder.csv"
    csv_path.write_text(
        "audio_path,recording_id\n"
        "subfolder/nested_audio.wav,\n",
        encoding="utf-8",
    )

    reader = ManifestReader(
        csv_path=csv_path,
        upload_dir=upload_dir,
        audio_path_col="audio_path",
        id_col="recording_id",
        allow_absolute=False,
    )

    rows = reader.load()
    assert len(rows) == 1
    assert rows[0].audio_path == (sub_dir / "nested_audio.wav").resolve()
    # Derived ID should replace separator with underscore
    assert rows[0].recording_id == "subfolder_nested_audio"


def test_count_data_rows_without_files(tmp_path: Path):
    """Test that count_data_rows works purely on CSV rows without checking audio files."""
    csv_path = tmp_path / "ghost_manifest.csv"
    csv_path.write_text(
        "audio_path,recording_id\n"
        "non_existent_1.wav,rec1\n"
        "non_existent_2.wav,rec2\n",
        encoding="utf-8",
    )

    reader = ManifestReader(
        csv_path=csv_path,
        upload_dir=tmp_path / "empty_dir",
        audio_path_col="audio_path",
        id_col="recording_id",
        allow_absolute=False,
    )

    assert reader.count_data_rows() == 2
