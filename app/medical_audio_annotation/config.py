"""Central configuration module for medical audio annotation pipeline."""

import csv
import logging
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Project directories
PROJECT_ROOT: Path = Path(__file__).resolve().parent
UPLOAD_DIR: Path = PROJECT_ROOT / "upload"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CSV_PATH: Path = UPLOAD_DIR / "audio_manifest.csv"

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# CSV Manifest Configuration
CSV_AUDIO_PATH_COLUMN: str = "audio_path"
CSV_ID_COLUMN: str = "recording_id"
ALLOWED_AUDIO_EXTENSIONS: set[str] = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
ALLOW_ABSOLUTE_AUDIO_PATHS: bool = False

# Batch Processing Range
START_INDEX: int = 0
END_INDEX: int | None = None  # None means "all remaining rows"


def count_csv_data_rows(path: Path) -> int:
    """Count non-header data rows in the CSV manifest file.

    If file does not exist, return 0 and emit a warning without crashing.
    """
    if not path.is_file():
        warnings.warn(f"CSV manifest file not found at {path}. ALL set to 0.", UserWarning)
        return 0
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
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
    except Exception as exc:
        warnings.warn(f"Failed to read CSV manifest at {path}: {exc}. ALL set to 0.", UserWarning)
        return 0


# Computed row count at module load time
ALL: int = count_csv_data_rows(CSV_PATH)


def compute_effective_end_index(end_idx: int | None, all_count: int) -> int:
    """Compute effective end index given end_index and total count."""
    if end_idx is None:
        return all_count
    return min(end_idx, all_count)


EFFECTIVE_END_INDEX: int = compute_effective_end_index(END_INDEX, ALL)

# OpenAI API Settings
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = "gpt-4o-audio-preview"
REQUEST_TIMEOUT_SECONDS: int = 120
MAX_RETRIES: int = 3
RETRY_BACKOFF_SECONDS: float = 2.0
MAX_CONCURRENT_REQUESTS: int = 2
MAX_AUDIO_BYTES: int = 25 * 1024 * 1024  # 25 MB
RESPONSE_FORMAT: str = "json"
OVERWRITE_RESULTS: bool = False
DRY_RUN: bool = False
CONTINUE_ON_ERROR: bool = True
SAVE_PREPROCESSED_AUDIO: bool = False
LOG_LEVEL: str = "INFO"

# Preprocessing Settings
TARGET_SAMPLE_RATE: int | None = None  # only set if required by inference method
TARGET_CHANNELS: int | None = None
NORMALIZE_AUDIO: bool = False
NORMALIZE_TARGET_DB: float = -3.0
TRIM_SILENCE: bool = False  # disabled: no robust timestamp-restoration mechanism
PREPROCESSING_VERSION: str = "1.0"


def validate_config(
    start_index: int | None = None,
    end_index: int | None = None,
    all_count: int | None = None,
    dry_run: bool | None = None,
    api_key: str | None = None,
    trim_silence: bool | None = None,
) -> int:
    """Validate pipeline configuration settings.

    Computes and returns the effective end index while validating:
    0 <= START_INDEX <= EFFECTIVE_END_INDEX <= ALL.

    Raises:
        ValueError: If batch indices, silence trimming, or required configurations are invalid.
    """
    global EFFECTIVE_END_INDEX

    start = START_INDEX if start_index is None else start_index
    end = END_INDEX if end_index is None else end_index
    total = ALL if all_count is None else all_count
    is_dry_run = DRY_RUN if dry_run is None else dry_run
    key = OPENAI_API_KEY if api_key is None else api_key
    silence = TRIM_SILENCE if trim_silence is None else trim_silence

    if silence:
        raise ValueError(
            "TRIM_SILENCE is enabled, but no timestamp restoration mechanism is implemented. Silence trimming is disabled."
        )

    if start < 0:
        raise ValueError(f"START_INDEX ({start}) cannot be negative.")

    if end is not None and end < 0:
        raise ValueError(f"END_INDEX ({end}) cannot be negative.")

    effective_end = compute_effective_end_index(end, total)

    if start > effective_end:
        raise ValueError(
            f"START_INDEX ({start}) exceeds EFFECTIVE_END_INDEX ({effective_end})."
        )

    if effective_end > total:
        raise ValueError(
            f"EFFECTIVE_END_INDEX ({effective_end}) exceeds ALL ({total})."
        )

    if not is_dry_run and not key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Provide it in .env or environment, or enable DRY_RUN."
        )

    EFFECTIVE_END_INDEX = effective_end
    return effective_end
