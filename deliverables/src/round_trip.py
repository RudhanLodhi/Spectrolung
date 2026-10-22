"""Model-free timing diagnostic for temporal downsample/upsample choices."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .labels import events


@dataclass(frozen=True)
class SweepResult:
    stride: int
    mode: str
    vanished_pct: float
    onset_mean_ms: float
    onset_median_ms: float
    onset_p90_ms: float
    offset_mean_ms: float
    offset_median_ms: float
    offset_p90_ms: float
    gap_survival_ge_150ms: float


def _quantize(labels: np.ndarray, stride: int, mode: str) -> np.ndarray:
    if mode not in {"left", "center"}:
        raise ValueError("mode must be 'left' or 'center'")
    n = len(labels)
    bins = (n + stride - 1) // stride
    padded = np.pad(labels, (0, bins * stride - n), constant_values=0)
    if mode == "left":
        sample = padded[::stride]
    else:
        sample = padded[stride // 2 :: stride]
        sample = np.pad(sample, (0, max(0, bins - len(sample))), constant_values=0)
    return np.repeat(sample[:bins], stride)[:n]


def _stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (float("nan"),) * 3
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(np.median(a)), float(np.percentile(a, 90))


def sweep(labels: np.ndarray, fps: float = 100.0, strides=(4, 8, 16, 32, 64)) -> pd.DataFrame:
    """Run the timing sweep against an integer frame-label vector."""
    labels = np.asarray(labels, dtype=np.int64)
    ground_truth = {c: events(labels, c, fps) for c in (1, 2)}
    rows: list[dict[str, object]] = []
    for stride in strides:
        for mode in ("left", "center"):
            reconstructed = _quantize(labels, stride, mode)
            predicted = {c: events(reconstructed, c, fps) for c in (1, 2)}
            onset: list[float] = []
            offset: list[float] = []
            total = vanished = 0
            for class_id in (1, 2):
                for start, end in ground_truth[class_id]:
                    total += 1
                    candidates = [
                        (abs(ps - start), ps, pe)
                        for ps, pe in predicted[class_id]
                        if pe > start and ps < end
                    ]
                    if not candidates:
                        vanished += 1
                        continue
                    _, ps, pe = min(candidates)
                    onset.append((ps - start) * 1000)
                    offset.append((pe - end) * 1000)
            gaps = []
            for class_id in (1, 2):
                spans = ground_truth[class_id]
                gaps.extend((b[0] - a[1]) for a, b in zip(spans, spans[1:]))
            kept = []
            for class_id in (1, 2):
                spans = ground_truth[class_id]
                for (_, end), (start, _) in zip(spans, spans[1:]):
                    if start - end < 0.15:
                        continue
                    lo, hi = int(np.ceil(end * fps)), int(np.floor(start * fps))
                    kept.append(bool(np.all(reconstructed[lo:hi] == 0)))
            on_mean, on_med, on_p90 = _stats(onset)
            off_mean, off_med, off_p90 = _stats(offset)
            rows.append({
                "stride": stride,
                "mode": mode,
                "vanished_pct": 100 * vanished / max(total, 1),
                "onset_mean_ms": on_mean,
                "onset_median_ms": on_med,
                "onset_p90_ms": on_p90,
                "offset_mean_ms": off_mean,
                "offset_median_ms": off_med,
                "offset_p90_ms": off_p90,
                "gap_survival_ge_150ms": float(np.mean(kept)) if kept else float("nan"),
            })
    return pd.DataFrame(rows)


def sweep_recordings(
    recordings: list[np.ndarray],
    fps: float = 100.0,
    strides=(4, 8, 16, 32, 64),
) -> pd.DataFrame:
    """Aggregate a sweep across recordings without joining unrelated timelines."""
    frames = [sweep(labels, fps, strides) for labels in recordings if len(labels)]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .groupby(["stride", "mode"], as_index=False)
        .mean(numeric_only=True)
    )
