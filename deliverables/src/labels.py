"""Annotation-to-frame conversion without modifying source annotations."""

from collections.abc import Iterable
import numpy as np

RAW2IDX = {"I": 1, "E": 2}


def make_frame_labels(
    rows: Iterable[tuple[float, float, str]],
    n_frames: int,
    fps: float = 100.0,
) -> np.ndarray:
    """Assign labels to frames centered at ``i / fps``.

    Non-phase labels remain background (0). Later overlapping phase rows win,
    matching the documented notebook behavior.
    """
    if n_frames < 0 or fps <= 0:
        raise ValueError("n_frames must be non-negative and fps must be positive")
    times = np.arange(n_frames, dtype=np.float64) / fps
    labels = np.zeros(n_frames, dtype=np.int64)
    for start, end, raw_label in rows:
        class_id = RAW2IDX.get(str(raw_label).strip())
        if class_id is None:
            continue
        if float(end) < float(start):
            raise ValueError(f"annotation end precedes start: {start}, {end}")
        mask = (times >= float(start)) & (times < float(end))
        labels[mask] = class_id
    return labels


def events(labels: np.ndarray, class_id: int, fps: float = 100.0) -> list[tuple[float, float]]:
    """Convert a frame-label vector into half-open second spans."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    active = np.asarray(labels) == class_id
    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(float(s / fps), float(e / fps)) for s, e in zip(starts, ends)]
