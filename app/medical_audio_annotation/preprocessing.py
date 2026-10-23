"""Audio preprocessing module for medical audio annotation pipeline."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from . import config
from .csv_reader import ManifestRow


@dataclass
class PreprocessingResult:
    """Stores metadata and paths for preprocessed audio."""

    recording_id: str
    source_path: Path
    prepared_path: Path  # Path to file ready for API submission (may equal source_path)
    original_sample_rate: int
    original_channels: int
    original_duration_seconds: float
    processed_duration_seconds: float
    sample_rate: int
    is_converted: bool  # True if a new file was created
    preprocessing_version: str


class AudioPreprocessor:
    """Preprocesses audio files according to configuration settings."""

    def __init__(
        self,
        target_sample_rate: Optional[int] = config.TARGET_SAMPLE_RATE,
        target_channels: Optional[int] = config.TARGET_CHANNELS,
        normalize_audio: bool = config.NORMALIZE_AUDIO,
        normalize_target_db: float = config.NORMALIZE_TARGET_DB,
        trim_silence: bool = config.TRIM_SILENCE,
        max_audio_bytes: int = config.MAX_AUDIO_BYTES,
        allowed_extensions: set[str] = config.ALLOWED_AUDIO_EXTENSIONS,
        save_preprocessed_audio: bool = config.SAVE_PREPROCESSED_AUDIO,
        results_dir: Path = config.RESULTS_DIR,
        preprocessing_version: str = config.PREPROCESSING_VERSION,
    ) -> None:
        """Initialize the audio preprocessor.

        Args:
            target_sample_rate: Target sample rate for resampling, or None to preserve.
            target_channels: Target number of channels (1=mono, 2=stereo), or None to preserve.
            normalize_audio: Whether to apply peak normalization.
            normalize_target_db: Target dBFS peak level for normalization (e.g., -3.0 dB).
            trim_silence: Whether silence trimming is requested (unsupported; raises error).
            max_audio_bytes: Maximum allowed file size in bytes.
            allowed_extensions: Set of permitted audio file extensions.
            save_preprocessed_audio: Whether to persist preprocessed files in results/_preprocessed.
            results_dir: Results directory path.
            preprocessing_version: Version tag of the preprocessing logic.
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.normalize_audio = normalize_audio
        self.normalize_target_db = normalize_target_db
        self.trim_silence = trim_silence
        self.max_audio_bytes = max_audio_bytes
        self.allowed_extensions = {ext.lower() for ext in allowed_extensions}
        self.save_preprocessed_audio = save_preprocessed_audio
        self.results_dir = Path(results_dir)
        self.preprocessing_version = preprocessing_version

    def process(self, row: ManifestRow) -> PreprocessingResult:
        """Process an audio file referenced in a manifest row.

        Reads original audio metadata, performs validation, and applies enabled
        timeline-preserving transformations (peak normalization, channel conversion,
        resampling).

        Note on duration:
            processed_duration_seconds strictly equals original_duration_seconds unless
            resampling is performed, in which case minor differences may arise from integer
            sample count rounding: (num_resampled_samples / target_sample_rate).

        Args:
            row: ManifestRow containing audio path and recording ID.

        Returns:
            PreprocessingResult containing source and prepared paths and metadata.

        Raises:
            ValueError: If file extension is unsupported, file size exceeds limit,
                silence trimming is enabled, or audio cannot be parsed.
        """
        source_path = row.audio_path

        # 1. Validate file extension
        ext = source_path.suffix.lower()
        if ext not in self.allowed_extensions:
            raise ValueError(
                f"Unsupported audio format/extension '{ext}' for file {source_path}. "
                f"Allowed extensions: {sorted(self.allowed_extensions)}"
            )

        # 2. Validate file size
        file_size = source_path.stat().st_size
        if file_size > self.max_audio_bytes:
            raise ValueError(
                f"Audio file '{source_path}' size ({file_size} bytes) exceeds "
                f"maximum limit of {self.max_audio_bytes} bytes."
            )

        # 3. Check silence trimming
        if self.trim_silence:
            raise ValueError(
                "TRIM_SILENCE is enabled, but no timestamp restoration mechanism is implemented. "
                "Silence trimming is disabled."
            )

        # 4. Read original metadata with soundfile
        try:
            sf_info = sf.info(str(source_path))
            orig_sr = sf_info.samplerate
            orig_channels = sf_info.channels
            orig_duration = float(sf_info.duration)
        except Exception as exc:
            raise ValueError(
                f"Failed to read audio file metadata with soundfile at '{source_path}': {exc}"
            ) from exc

        # 5. Check if any transformation is needed
        needs_resample = (
            self.target_sample_rate is not None and self.target_sample_rate != orig_sr
        )
        needs_channel_change = (
            self.target_channels is not None and self.target_channels != orig_channels
        )
        needs_normalization = self.normalize_audio

        if not (needs_resample or needs_channel_change or needs_normalization):
            return PreprocessingResult(
                recording_id=row.recording_id,
                source_path=source_path,
                prepared_path=source_path,
                original_sample_rate=orig_sr,
                original_channels=orig_channels,
                original_duration_seconds=orig_duration,
                processed_duration_seconds=orig_duration,
                sample_rate=orig_sr,
                is_converted=False,
                preprocessing_version=self.preprocessing_version,
            )

        # 6. Apply transformations
        try:
            data, sr = sf.read(str(source_path), dtype="float32")
        except Exception as exc:
            raise ValueError(
                f"Failed to read audio data from '{source_path}': {exc}"
            ) from exc

        # Channel adjustment
        if needs_channel_change and self.target_channels is not None:
            if self.target_channels == 1 and data.ndim > 1:
                data = np.mean(data, axis=1)
            elif self.target_channels == 2 and data.ndim == 1:
                data = np.column_stack([data, data])

        # Resampling
        target_sr = self.target_sample_rate if needs_resample and self.target_sample_rate else orig_sr
        if needs_resample and self.target_sample_rate is not None:
            from scipy.signal import resample

            num_samples = int(round(len(data) * float(target_sr) / orig_sr))
            data = resample(data, num_samples, axis=0)
            processed_duration = float(num_samples) / float(target_sr)
        else:
            processed_duration = orig_duration

        # Peak normalization (preserves duration)
        if needs_normalization:
            target_linear_peak = 10.0 ** (self.normalize_target_db / 20.0)
            current_peak = float(np.max(np.abs(data)))
            if current_peak > 0:
                scale = target_linear_peak / current_peak
                data = data * scale

        # Save preprocessed audio to temporary file or results directory
        if self.save_preprocessed_audio:
            prep_dir = self.results_dir / "_preprocessed"
            prep_dir.mkdir(parents=True, exist_ok=True)
            prepared_path = prep_dir / f"{row.recording_id}_prep.wav"
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            prepared_path = Path(tmp.name)
            tmp.close()

        sf.write(str(prepared_path), data, target_sr)

        return PreprocessingResult(
            recording_id=row.recording_id,
            source_path=source_path,
            prepared_path=prepared_path,
            original_sample_rate=orig_sr,
            original_channels=orig_channels,
            original_duration_seconds=orig_duration,
            processed_duration_seconds=processed_duration,
            sample_rate=target_sr,
            is_converted=True,
            preprocessing_version=self.preprocessing_version,
        )
