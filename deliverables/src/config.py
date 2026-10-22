"""Central configuration for the span-model pipeline."""

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass(frozen=True)
class PipelineConfig:
    sample_rate: int = 4000
    fps: int = 100
    hop_length: int = 40
    n_fft: int = 512
    n_mels: int = 64
    fmin: int = 25
    fmax: int = 2000
    encoder_time_stride: int = 4
    median_filter_ms: int = 120
    minimum_event_seconds: float = 0.15
    seed: int = 0
    cache_version: str = "s2s-v1"

    def validate(self) -> None:
        if self.sample_rate <= 0 or self.fps <= 0:
            raise ValueError("sample_rate and fps must be positive")
        if self.sample_rate % self.fps != 0:
            raise ValueError("sample_rate must be divisible by fps")
        if self.hop_length != self.sample_rate // self.fps:
            raise ValueError("hop_length must equal sample_rate // fps")
        if not 1 <= self.encoder_time_stride <= 64:
            raise ValueError("encoder_time_stride must be in [1, 64]")
        if not 0 < self.fmin < self.fmax <= self.sample_rate // 2:
            raise ValueError("frequency range must fit below Nyquist")

    def to_json(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
