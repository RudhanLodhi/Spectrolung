# Medical Audio Annotation Pipeline

A Python pipeline for automated medical audio annotation using audio-capable large language models (OpenAI `gpt-4o-audio-preview`). The pipeline ingests medical acoustic recordings, performs safe audio preprocessing and validation, prompts the model for timestamped acoustic event detection, and atomically serializes structured JSON results.

> **CRITICAL CLINICAL & REGULATORY DISCLAIMER:**
> This software is strictly an experimental research tool for acoustic signal annotation and characterization. It is **NOT** a medical device, is **NOT** FDA/CE approved or cleared, and must **NEVER** be used for clinical diagnosis, patient management, or clinical decision-making. No medical diagnosis should ever be inferred solely from model-generated acoustic annotations.

---

## 1. Prerequisites and Installation

### Prerequisites
- Python 3.10+
- An OpenAI API key with access to `gpt-4o-audio-preview`

### Installation
From the package directory (`c:\Melspectrolungs\app\medical_audio_annotation`):

```bash
pip install -r requirements.txt
```

### Environment Configuration
Copy `.env.example` to `.env` and set your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 2. Configuration Settings

All pipeline options are centralized in `config.py`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `PROJECT_ROOT` | `Path` | Directory of `config.py` | Root directory of the annotation module |
| `UPLOAD_DIR` | `Path` | `PROJECT_ROOT / "upload"` | Directory containing source audio files |
| `RESULTS_DIR` | `Path` | `PROJECT_ROOT / "results"` | Directory where JSON results are saved |
| `CSV_PATH` | `Path` | `UPLOAD_DIR / "audio_manifest.csv"` | Path to the manifest file |
| `CSV_AUDIO_PATH_COLUMN` | `str` | `"audio_path"` | Column name for audio file paths |
| `CSV_ID_COLUMN` | `str` | `"recording_id"` | Column name for recording identifiers |
| `ALLOWED_AUDIO_EXTENSIONS`| `set` | `{".wav", ".mp3", ...}` | Permitted audio file extensions |
| `ALLOW_ABSOLUTE_AUDIO_PATHS`| `bool` | `False` | Disallow absolute paths to guard against path traversal |
| `START_INDEX` | `int` | `0` | Starting index in manifest batch |
| `END_INDEX` | `int \| None` | `None` | Ending index in manifest batch (`None` = all remaining) |
| `OPENAI_MODEL` | `str` | `"gpt-4o-audio-preview"` | Audio-capable model name |
| `REQUEST_TIMEOUT_SECONDS` | `int` | `120` | HTTP request timeout |
| `MAX_RETRIES` | `int` | `3` | Maximum retry attempts for transient API errors |
| `RETRY_BACKOFF_SECONDS` | `float` | `2.0` | Base exponential backoff factor |
| `MAX_AUDIO_BYTES` | `int` | `26214400` (25 MB) | Maximum permitted audio file size |
| `OVERWRITE_RESULTS` | `bool` | `False` | Skip already annotated valid files unless True |
| `DRY_RUN` | `bool` | `False` | Run full flow without calling the OpenAI API |
| `CONTINUE_ON_ERROR` | `bool` | `True` | Continue processing remaining items if one fails |
| `SAVE_PREPROCESSED_AUDIO` | `bool` | `False` | Save converted/normalized audio in `results/_preprocessed/` |
| `NORMALIZE_AUDIO` | `bool` | `False` | Apply peak normalization (preserves duration) |
| `NORMALIZE_TARGET_DB` | `float` | `-3.0` | Target dBFS peak normalization level |
| `TRIM_SILENCE` | `bool` | `False` | Disabled: prevents corrupting temporal alignment |

---

## 3. CSV Manifest Format

The manifest must be a UTF-8 encoded CSV located at `upload/audio_manifest.csv`:

```csv
audio_path,recording_id
recording_001.wav,recording_001
recording_002.wav,recording_002
subfolder/recording_003.wav,recording_003
```

- `audio_path`: Relative path within `upload/`. Nested folders are supported and will avoid collisions in `results/`.
- `recording_id`: Optional identifier. If omitted, a stable ID is automatically derived from the audio path.
- Additional metadata columns (e.g. `patient_id`, `site`) are preserved in `extra_fields`.

---

## 4. Running the Pipeline

### Dry Run (Recommended for testing setup without API charges)
In `config.py`, set:
```python
DRY_RUN = True
```

Run:
```bash
python -m app.medical_audio_annotation.main
```

### Live Run
Ensure `DRY_RUN = False` and `OPENAI_API_KEY` is populated, then:
```bash
python -m app.medical_audio_annotation.main
```

---

## 5. Output Schema

Results are written atomically as JSON files to `results/{recording_id}.json`. If recordings originate from nested folders, matching subdirectories are created under `results/`.

### Schema Definition

```json
{
  "schema_version": "1.0",
  "recording_id": "recording_001",
  "source_file": "c:\\Melspectrolungs\\app\\medical_audio_annotation\\upload\\recording_001.wav",
  "status": "success",
  "model": "gpt-4o-audio-preview",
  "preprocessing_version": "1.0",
  "original_duration_seconds": 4.500,
  "events": [
    {
      "event_id": "evt_01",
      "label": "provisional_wheeze",
      "description": "Continuous high-pitched musical sound during expiration",
      "start_seconds": 1.25,
      "end_seconds": 2.40,
      "uncertain": false,
      "uncertainty_reason": null,
      "confidence_score": 0.88
    }
  ],
  "error": null
}
```

### Fields
- `status`: `"success"`, `"dry_run"`, or `"error"`.
- `events`: Array of temporal acoustic events with `start_seconds` and `end_seconds` relative to original audio timeline.
- `uncertain`: Boolean flag indicating whether acoustic evidence is ambiguous or degraded by noise.
- `error`: Populated with `{"code": str, "message": str}` when status is `"error"`. Sensitive credential patterns like API keys are automatically redacted.

---

## 6. Running Tests

Unit and integration tests use `pytest` and mock external network calls:

```bash
python -m pytest tests/ -v
```

---

## 7. Limitations & Clinical Disclaimer

1. **Research Use Only**: Designed for exploratory audio labeling and benchmark generation. Not certified for diagnostic or patient monitoring use.
2. **Provisional Ontologies**: Outputs use provisional acoustic labels (e.g. `provisional_wheeze`, `provisional_crackles`) rather than clinical disease tags.
3. **Uncalibrated Model Confidence**: Self-assessed model confidence scores are heuristic and must not be interpreted as validated posterior probabilities.
4. **Silence Trimming**: Silence trimming is disabled (`TRIM_SILENCE = False`) to prevent altering timestamps relative to the original source recordings.
