"""CSV Manifest reader and row validator for medical audio annotation."""

import csv
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath, PurePosixPath


@dataclass
class ManifestRow:
    """Represents a validated row from the audio manifest CSV."""

    row_index: int
    recording_id: str
    audio_path: Path
    extra_fields: dict[str, str] = field(default_factory=dict)


class ManifestReader:
    """Reads and validates the audio manifest CSV against the upload directory."""

    def __init__(
        self,
        csv_path: Path,
        upload_dir: Path,
        audio_path_col: str = "audio_path",
        id_col: str = "recording_id",
        allow_absolute: bool = False,
    ) -> None:
        """Initialize the manifest reader with paths and configuration.

        Args:
            csv_path: Path to the CSV manifest file.
            upload_dir: Directory containing audio files.
            audio_path_col: Name of column containing audio file paths.
            id_col: Name of column containing recording IDs.
            allow_absolute: Whether to allow absolute audio file paths.
        """
        self.csv_path = Path(csv_path)
        self.upload_dir = Path(upload_dir)
        self.audio_path_col = audio_path_col
        self.id_col = id_col
        self.allow_absolute = allow_absolute

    def count_data_rows(self) -> int:
        """Count non-header data rows without verifying audio file existence.

        Returns:
            Number of non-empty data rows.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Manifest CSV not found: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = None
            for row in reader:
                if any(cell.strip() for cell in row):
                    header = row
                    break
            if header is None:
                return 0
            count = 0
            for row in reader:
                if any(cell.strip() for cell in row):
                    count += 1
            return count

    def load(self) -> list[ManifestRow]:
        """Read and validate all manifest rows.

        Returns:
            List of validated ManifestRow objects.

        Raises:
            FileNotFoundError: If CSV or any referenced audio file is missing.
            ValueError: On missing columns, malformed data, path traversal, or duplicates.
        """
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Manifest CSV not found: {self.csv_path}")

        upload_dir_resolved = self.upload_dir.resolve()
        rows: list[ManifestRow] = []
        seen_ids: set[str] = set()

        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Malformed or empty CSV file: {self.csv_path}")

            fieldnames = [fn.strip() for fn in reader.fieldnames if fn]
            if self.audio_path_col not in fieldnames:
                raise ValueError(
                    f"Missing required audio path column '{self.audio_path_col}' in CSV: {fieldnames}"
                )

            for idx, raw_row in enumerate(reader, start=1):
                # Clean keys and values
                row_data = {
                    (k.strip() if k else ""): (v.strip() if v else "")
                    for k, v in raw_row.items()
                }

                # Skip completely blank lines
                if not any(row_data.values()):
                    continue

                raw_audio_path = row_data.get(self.audio_path_col, "")
                if not raw_audio_path:
                    raise ValueError(
                        f"Row {idx}: Empty audio path in column '{self.audio_path_col}'."
                    )

                # Check absolute paths
                is_abs = (
                    Path(raw_audio_path).is_absolute()
                    or raw_audio_path.startswith(("/", "\\"))
                    or (len(raw_audio_path) > 1 and raw_audio_path[1] == ":")
                )
                if is_abs and not self.allow_absolute:
                    raise ValueError(
                        f"Row {idx}: Absolute audio path not allowed: '{raw_audio_path}'"
                    )

                # Resolve path and guard against traversal
                if is_abs:
                    resolved_audio_path = Path(raw_audio_path).resolve()
                else:
                    resolved_audio_path = (self.upload_dir / raw_audio_path).resolve()
                    try:
                        resolved_audio_path.relative_to(upload_dir_resolved)
                    except ValueError:
                        raise ValueError(
                            f"Row {idx}: Path traversal detected for '{raw_audio_path}' outside '{self.upload_dir}'"
                        )

                if not resolved_audio_path.is_file():
                    raise FileNotFoundError(
                        f"Row {idx}: Audio file not found at '{resolved_audio_path}'"
                    )

                # Determine recording ID
                explicit_id = row_data.get(self.id_col, "")
                if explicit_id:
                    rec_id = explicit_id
                else:
                    # Derive stable ID from relative path without extension
                    rel_p = Path(raw_audio_path)
                    rel_no_ext = rel_p.with_suffix("")
                    rec_id = (
                        str(rel_no_ext)
                        .replace("/", "_")
                        .replace("\\", "_")
                        .strip("_")
                    )
                    if not rec_id:
                        rec_id = rel_p.stem

                if rec_id in seen_ids:
                    raise ValueError(
                        f"Row {idx}: Duplicate recording_id encountered: '{rec_id}'"
                    )
                seen_ids.add(rec_id)

                # Extract extra fields
                extra_fields = {
                    k: v
                    for k, v in row_data.items()
                    if k not in (self.audio_path_col, self.id_col)
                }

                rows.append(
                    ManifestRow(
                        row_index=idx,
                        recording_id=rec_id,
                        audio_path=resolved_audio_path,
                        extra_fields=extra_fields,
                    )
                )

        return rows

    def get_batch(self, start: int, end: int) -> list[ManifestRow]:
        """Retrieve a slice of manifest rows [start:end].

        Args:
            start: Start index (0-based inclusive).
            end: End index (0-based exclusive).

        Returns:
            List of ManifestRow objects in the specified index range.
        """
        all_rows = self.load()
        return all_rows[start:end]
