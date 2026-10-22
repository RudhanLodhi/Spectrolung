"""HF_Lung inventory and discrepancy report."""

from pathlib import Path
import re
import pandas as pd

_STETH = re.compile(r"^steth_(\d{8})_")
_TRUNC = re.compile(r"^trunc_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-")


def patient_key(stem: str) -> str | None:
    match = _STETH.match(stem)
    if match:
        return f"steth:{match.group(1)}"
    match = _TRUNC.match(stem)
    return f"trunc:{match.group(1)}" if match else None


def build_inventory(data_root: Path) -> dict[str, object]:
    audio = sorted(data_root.glob("train/*.wav")) + sorted(data_root.glob("test/*.wav"))
    annotations_path = data_root / "annotations.csv"
    annotations = pd.read_csv(annotations_path) if annotations_path.exists() else pd.DataFrame()
    audio_names = {path.name for path in audio}
    annotation_names = set(annotations["filename"].astype(str)) if "filename" in annotations else set()
    sample_rates: dict[str, int] = {}
    try:
        import soundfile as sf
        for path in audio:
            info = sf.info(path)
            sample_rates[str(info.samplerate)] = sample_rates.get(str(info.samplerate), 0) + 1
    except ImportError:
        sample_rates = {"unavailable": len(audio)}
    return {
        "data_root": str(data_root),
        "audio_count": len(audio),
        "train_count": sum(path.parent.name == "train" for path in audio),
        "test_count": sum(path.parent.name == "test" for path in audio),
        "sample_rates": sample_rates,
        "annotation_rows": len(annotations),
        "audio_without_annotations": sorted(audio_names - annotation_names),
        "annotations_without_audio": sorted(annotation_names - audio_names),
        "unparseable_patient_ids": sorted(path.name for path in audio if patient_key(path.stem) is None),
        "phase_label_counts": (
            annotations[annotations["label"].isin(["I", "E"])]["label"].value_counts().to_dict()
            if "label" in annotations else {}
        ),
    }


def write_report(inventory: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HF_Lung inventory",
        "",
        "This report is generated from the current data directory; source files are read-only.",
        "",
    ]
    for key, value in inventory.items():
        if isinstance(value, list):
            lines.append(f"## {key.replace('_', ' ').title()} ({len(value)})")
            lines.extend(f"- `{item}`" for item in value[:100])
            if len(value) > 100:
                lines.append(f"- ... {len(value) - 100} more")
        else:
            lines.append(f"- **{key.replace('_', ' ').title()}:** `{value}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
