"""Main entry point and orchestrator for medical audio annotation pipeline."""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

from . import config
from .csv_reader import ManifestReader
from .inference import OpenAIAudioAnnotator
from .preprocessing import AudioPreprocessor, PreprocessingResult
from .prompt import SYSTEM_PROMPT, build_user_message
from .result_writer import AnnotationResult, ResultWriter

logger = logging.getLogger("medical_audio_annotation")


class PipelineRunner:
    """Orchestrates the loading, preprocessing, inference, and serialization pipeline."""

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        upload_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        dry_run: Optional[bool] = None,
        overwrite: Optional[bool] = None,
        continue_on_error: Optional[bool] = None,
    ) -> None:
        """Initialize pipeline runner with configurable paths and flags."""
        self.csv_path = Path(csv_path) if csv_path else config.CSV_PATH
        self.upload_dir = Path(upload_dir) if upload_dir else config.UPLOAD_DIR
        self.results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
        self.dry_run = config.DRY_RUN if dry_run is None else dry_run
        self.overwrite = config.OVERWRITE_RESULTS if overwrite is None else overwrite
        self.continue_on_error = (
            config.CONTINUE_ON_ERROR if continue_on_error is None else continue_on_error
        )

    def run(self) -> dict[str, Any]:
        """Execute the audio annotation pipeline.

        Returns:
            Dictionary with execution statistics.
        """
        start_time = time.time()

        # 1. Setup logging
        log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logger.setLevel(log_level)

        # 2. Count rows and validate configuration
        manifest_reader = ManifestReader(
            csv_path=self.csv_path,
            upload_dir=self.upload_dir,
            audio_path_col=config.CSV_AUDIO_PATH_COLUMN,
            id_col=config.CSV_ID_COLUMN,
            allow_absolute=config.ALLOW_ABSOLUTE_AUDIO_PATHS,
        )

        all_count = manifest_reader.count_data_rows()
        effective_end = config.validate_config(
            start_index=config.START_INDEX,
            end_index=config.END_INDEX,
            all_count=all_count,
            dry_run=self.dry_run,
            api_key=config.OPENAI_API_KEY,
            trim_silence=config.TRIM_SILENCE,
        )

        batch = manifest_reader.get_batch(config.START_INDEX, effective_end)

        # 3. Print startup banner
        print("=" * 60)
        print("Medical Audio Annotation Pipeline")
        print("=" * 60)
        print(f"Manifest rows:   {all_count}")
        print(f"Selected range:  [{config.START_INDEX}:{effective_end}]")
        print(f"Selected count:  {len(batch)}")
        print(f"Mode:            {'DRY RUN' if self.dry_run else 'LIVE'}")
        print("=" * 60)

        # 4. Initialize components
        preprocessor = AudioPreprocessor(
            target_sample_rate=config.TARGET_SAMPLE_RATE,
            target_channels=config.TARGET_CHANNELS,
            normalize_audio=config.NORMALIZE_AUDIO,
            normalize_target_db=config.NORMALIZE_TARGET_DB,
            trim_silence=config.TRIM_SILENCE,
            max_audio_bytes=config.MAX_AUDIO_BYTES,
            allowed_extensions=config.ALLOWED_AUDIO_EXTENSIONS,
            save_preprocessed_audio=config.SAVE_PREPROCESSED_AUDIO,
            results_dir=self.results_dir,
            preprocessing_version=config.PREPROCESSING_VERSION,
        )

        annotator = OpenAIAudioAnnotator(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            timeout=config.REQUEST_TIMEOUT_SECONDS,
            max_retries=config.MAX_RETRIES,
            backoff=config.RETRY_BACKOFF_SECONDS,
            dry_run=self.dry_run,
        )

        result_writer = ResultWriter(
            results_dir=self.results_dir,
            overwrite=self.overwrite,
            upload_dir=self.upload_dir,
        )

        succeeded_count = 0
        skipped_count = 0
        failed_count = 0

        # 5. Process each row in the batch
        for row in batch:
            result_path = result_writer.get_result_path(row)

            # Skip if valid result exists and overwrite disabled
            if not self.overwrite and result_writer.is_valid_result(result_path):
                logger.info(
                    "Skipping %s: valid result already exists at %s",
                    row.recording_id,
                    result_path,
                )
                skipped_count += 1
                continue

            prep_result: Optional[PreprocessingResult] = None
            try:
                # Preprocess audio
                prep_result = preprocessor.process(row)

                # Build prompt
                user_message = build_user_message(
                    recording_id=row.recording_id,
                    duration_seconds=prep_result.processed_duration_seconds,
                )

                # Call inference
                response = annotator.annotate(
                    result=prep_result,
                    system_prompt=SYSTEM_PROMPT,
                    user_message=user_message,
                )

                # Assemble result
                status = "dry_run" if self.dry_run else "success"
                annotation_result = AnnotationResult(
                    schema_version="1.0",
                    recording_id=row.recording_id,
                    source_file=str(row.audio_path),
                    status=status,
                    model=response.model,
                    preprocessing_version=prep_result.preprocessing_version,
                    original_duration_seconds=prep_result.original_duration_seconds,
                    events=response.events,
                    error=None,
                )

                # Persist result
                written_path = result_writer.write(annotation_result, row)
                logger.info("Wrote result for %s to %s", row.recording_id, written_path)
                succeeded_count += 1

            except Exception as exc:
                logger.exception(
                    "Error processing recording %s: %s", row.recording_id, exc
                )
                failed_count += 1
                orig_dur = (
                    prep_result.original_duration_seconds if prep_result else None
                )
                result_writer.write_error(
                    row=row,
                    code=type(exc).__name__,
                    message=str(exc),
                    original_duration_seconds=orig_dur,
                )
                if not self.continue_on_error:
                    raise

        elapsed = time.time() - start_time

        # 6. Print final summary
        print("\n" + "=" * 60)
        print("Pipeline Execution Summary")
        print("=" * 60)
        print(f"Manifest rows:     {all_count}")
        print(f"Selected range:    [{config.START_INDEX}:{effective_end}]")
        print(f"Selected rows:     {len(batch)}")
        print(f"Succeeded:         {succeeded_count}")
        print(f"Skipped:           {skipped_count}")
        print(f"Failed:            {failed_count}")
        print(f"Elapsed time:      {elapsed:.2f}s")
        print(f"Results directory: {self.results_dir}")
        print("=" * 60)

        return {
            "manifest_rows": all_count,
            "selected_range": (config.START_INDEX, effective_end),
            "selected_rows": len(batch),
            "succeeded": succeeded_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "elapsed_seconds": elapsed,
            "results_dir": self.results_dir,
        }


def main() -> None:
    """Run pipeline as CLI command."""
    runner = PipelineRunner()
    runner.run()


if __name__ == "__main__":
    main()
