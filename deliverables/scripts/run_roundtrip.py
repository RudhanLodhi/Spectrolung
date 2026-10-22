from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labels import make_frame_labels
from src.round_trip import sweep_recordings

parser = ArgumentParser()
parser.add_argument("--annotations", type=Path, default=Path("HF_Lung/annotations.csv"))
parser.add_argument("--output", type=Path, default=ROOT / "results" / "02_roundtrip_sweep.csv")
parser.add_argument("--plot", type=Path, default=ROOT / "results" / "02_roundtrip_sweep.png")
parser.add_argument("--fps", type=float, default=100.0)
args = parser.parse_args()

annotations = pd.read_csv(args.annotations)
recordings = []
for _, group in annotations.groupby("filename", sort=False):
    rows = [
        (row.start_time, row.end_time, row.label)
        for row in group.itertuples()
        if row.label in {"I", "E"}
    ]
    if not rows:
        continue
    n_frames = int((group["end_time"].max() * args.fps) + 1)
    recordings.append(make_frame_labels(rows, n_frames, args.fps))
result = sweep_recordings(recordings, args.fps)
args.output.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(args.output, index=False)
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode, group in result.groupby("mode"):
        axes[0].plot(group["stride"], group["onset_median_ms"], marker="o", label=mode)
        axes[1].plot(group["stride"], group["gap_survival_ge_150ms"], marker="o", label=mode)
    axes[0].set(xlabel="Temporal stride", ylabel="Median onset error (ms)")
    axes[1].set(xlabel="Temporal stride", ylabel="Gap survival (>=150 ms)")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.savefig(args.plot, dpi=160)
    plt.close(fig)
except ImportError:
    print("matplotlib is unavailable; CSV was still generated")
print(args.output)
