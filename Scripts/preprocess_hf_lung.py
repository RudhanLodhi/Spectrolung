"""
=============================================================================
  HF-Lung V1 Dataset Preprocessing Script
  Restructures the raw HF_lung_V1 dataset into a standardized format.
=============================================================================

Output structure:
  HF_Lung/
    Audio Files/        <- all .wav files copied over
    Patient_info.csv    <- dynamic patient registry (S-IDs and T-IDs)
    Events.csv          <- parsed & merged breath cycles with labels
"""

import re
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE    = Path(r"C:\Spectrolung")
RAW_DIR = BASE / "HF_lung_V1"
OUT_DIR = BASE / "HF_Lung"
AUD_OUT = OUT_DIR / "Audio Files"

OUT_DIR.mkdir(exist_ok=True)
AUD_OUT.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Position Mapping (L1-L8 anatomical names)
# ─────────────────────────────────────────────────────────────────────────────
POSITION_MAP = {
    "L1": "Right Anterior Superior (2nd ICS, Right MCL)",
    "L2": "Right Anterior Inferior (5th ICS, Right MCL)",
    "L3": "Right Lateral Superior (4th ICS, Right MAL)",
    "L4": "Right Lateral Inferior (10th ICS, Right MAL)",
    "L5": "Left Anterior Superior (2nd ICS, Left MCL)",
    "L6": "Left Anterior Inferior (5th ICS, Left MCL)",
    "L7": "Left Lateral Superior (4th ICS, Left MAL)",
    "L8": "Left Lateral Inferior (10th ICS, Left MAL)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Adventitious sound label mapping
# ─────────────────────────────────────────────────────────────────────────────
SOUND_MAP = {
    "D":       "Coarse Crackle",
    "Wheeze":  "Wheeze",
    "W":       "Wheeze",
    "C":       "Wheeze",
    "Rhonchi": "Rhonchi",
    "Rhonchus":"Rhonchi",
    "R":       "Rhonchi",
    "Stridor": "Stridor",
    "S":       "Stridor",
}

# Phase labels (not adventitious sounds)
PHASE_LABELS = {"I", "E"}


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp parser
# ─────────────────────────────────────────────────────────────────────────────
_TS_RE = re.compile(r"(\d+):(\d+):([\d.]+)")

def _ts_to_s(ts: str) -> float:
    m = _TS_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Cannot parse timestamp: {ts!r}")
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s


# ─────────────────────────────────────────────────────────────────────────────
# Filename parsing
# ─────────────────────────────────────────────────────────────────────────────
# steth_20181210_08_50_03.wav
_STETH_RE = re.compile(r"steth_(\d{4})(\d{2})(\d{2})_(\d{2})_(\d{2})_(\d{2})")
# trunc_2019-07-31-11-20-43-L8_12.wav
_TRUNC_RE = re.compile(r"trunc_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(L\d+)_(\d+)")


class PatientRegistry:
    """Auto-incrementing patient ID registry."""
    def __init__(self, prefix: str):
        self.prefix = prefix
        self._map: OrderedDict[str, str] = OrderedDict()
        self._counter = 0

    def get_or_create(self, key: str) -> str:
        if key not in self._map:
            self._counter += 1
            self._map[key] = f"{self.prefix}{self._counter}"
        return self._map[key]

    def all_entries(self) -> list[dict]:
        return [{"Patient_ID": pid, "Session_Key": key}
                for key, pid in self._map.items()]

    def __len__(self):
        return len(self._map)


def parse_steth_filename(stem: str, registry: PatientRegistry) -> dict | None:
    m = _STETH_RE.match(stem)
    if not m:
        return None
    y, mo, d, hh, mm, ss = m.groups()
    date_key = f"{y}{mo}{d}"
    return {
        "Patient_ID":       registry.get_or_create(date_key),
        "Date":             date_key,
        "Recording_Time":   f"{hh}:{mm}:{ss}",
        "Position":         "Unknown",
        "Chunk_Index":      "Unknown",
        "Equipment":        "Littmann 3200",
        "Acquisition_Mode": "sc",
    }


def parse_trunc_filename(stem: str, registry: PatientRegistry) -> dict | None:
    m = _TRUNC_RE.match(stem)
    if not m:
        return None
    y, mo, d, hh, mm, ss, loc, chunk = m.groups()
    session_key = f"{y}-{mo}-{d}-{hh}-{mm}-{ss}"
    raw_pos = loc
    return {
        "Patient_ID":       registry.get_or_create(session_key),
        "Date":             f"{y}{mo}{d}",
        "Recording_Time":   f"{hh}:{mm}:{ss}",
        "Position":         POSITION_MAP.get(raw_pos, raw_pos),
        "Position_Code":    raw_pos,
        "Chunk_Index":      int(chunk),
        "Equipment":        "HF-Type-1",
        "Acquisition_Mode": "mc",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase-merging engine
# ─────────────────────────────────────────────────────────────────────────────
MAX_GAP = 1.5  # seconds: max gap between I-end and E-start to merge

def _overlaps(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def build_cycles_and_label(label_file: Path) -> list[dict]:
    """
    Parse a _label.txt file, merge I+E into breath cycles, and label
    each cycle based on overlapping adventitious sounds.
    """
    inspirations = []
    expirations  = []
    adventitious = []  # (start, end, mapped_label)

    with open(label_file, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 3:
                continue
            ev_label = parts[0]
            try:
                t0 = _ts_to_s(parts[1])
                t1 = _ts_to_s(parts[2])
            except ValueError:
                continue

            if ev_label == "I":
                inspirations.append((t0, t1))
            elif ev_label == "E":
                expirations.append((t0, t1))
            elif ev_label in SOUND_MAP:
                adventitious.append((t0, t1, SOUND_MAP[ev_label]))

    if not inspirations and not expirations:
        return []

    inspirations.sort(key=lambda x: x[0])
    expirations.sort(key=lambda x: x[0])

    # ── Merge I + E into cycles ──
    # Strategy: for each I, look for the next E that starts within MAX_GAP
    # of the I's end. If found, merge them into one cycle. Mark used E's.
    # Any leftover I's or E's become standalone cycles.
    used_e = set()
    cycles = []

    for i_start, i_end in inspirations:
        merged = False
        for j, (e_start, e_end) in enumerate(expirations):
            if j in used_e:
                continue
            gap = e_start - i_end
            if -0.15 <= gap <= MAX_GAP:
                # Merge: cycle spans from I-start to E-end
                cycles.append((i_start, e_end))
                used_e.add(j)
                merged = True
                break
            elif e_start > i_end + MAX_GAP:
                # Past the gap window, stop searching
                break
        if not merged:
            # Isolated inspiration -> treat as standalone cycle
            cycles.append((i_start, i_end))

    # Any unused expirations become standalone cycles
    for j, (e_start, e_end) in enumerate(expirations):
        if j not in used_e:
            cycles.append((e_start, e_end))

    cycles.sort(key=lambda x: x[0])

    # ── Label each cycle ──
    results = []
    for c_start, c_end in cycles:
        # Find all adventitious sounds overlapping this cycle
        hits = set()
        for a_start, a_end, a_label in adventitious:
            if _overlaps(c_start, c_end, a_start, a_end):
                hits.add(a_label)

        if not hits:
            label = "Normal"
        else:
            # Sort for consistent ordering: Wheeze, Coarse Crackle, Rhonchi, Stridor
            priority = ["Wheeze", "Coarse Crackle", "Rhonchi", "Stridor"]
            sorted_hits = [h for h in priority if h in hits]
            # Add any we might have missed (future-proofing)
            for h in sorted(hits):
                if h not in sorted_hits:
                    sorted_hits.append(h)
            label = "+".join(sorted_hits)

        results.append({
            "cycle_start": round(c_start, 3),
            "cycle_end":   round(c_end, 3),
            "Label":       label,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 62)
    print("  HF-LUNG V1 PREPROCESSING")
    print("=" * 62 + "\n")

    steth_registry = PatientRegistry("S")
    trunc_registry = PatientRegistry("T")

    all_records = []
    n_copied = 0
    n_skipped_copy = 0

    for split in ["train", "test"]:
        split_dir = RAW_DIR / split
        label_files = sorted(split_dir.glob("*_label.txt"))
        print(f"[{split}] Found {len(label_files)} label files")

        for label_file in tqdm(label_files, desc=f"  {split}", unit="file"):
            stem = label_file.stem.replace("_label", "")
            wav_file = split_dir / f"{stem}.wav"
            if not wav_file.exists():
                continue

            # ── Copy audio file ──
            dest = AUD_OUT / wav_file.name
            if not dest.exists():
                shutil.copy2(wav_file, dest)
                n_copied += 1
            else:
                n_skipped_copy += 1

            # ── Parse filename ──
            if stem.startswith("steth_"):
                meta = parse_steth_filename(stem, steth_registry)
            elif stem.startswith("trunc_"):
                meta = parse_trunc_filename(stem, trunc_registry)
            else:
                # Unknown prefix, skip
                continue

            if meta is None:
                continue

            recording_id = stem

            # ── Build cycles & labels ──
            cycles = build_cycles_and_label(label_file)

            for cyc in cycles:
                record = {
                    "Recording_ID":     recording_id,
                    "Patient_ID":       meta["Patient_ID"],
                    "Recording_Time":   meta["Recording_Time"],
                    "Position":         meta["Position"],
                    "Chunk_Index":      meta["Chunk_Index"],
                    "Equipment":        meta["Equipment"],
                    "Acquisition_Mode": meta["Acquisition_Mode"],
                    "cycle_start":      cyc["cycle_start"],
                    "cycle_end":        cyc["cycle_end"],
                    "Label":            cyc["Label"],
                }
                all_records.append(record)

    # ─────────────────────────────────────────────────────────────────────
    # Save Events.csv
    # ─────────────────────────────────────────────────────────────────────
    events_df = pd.DataFrame(all_records)
    events_cols = ["Recording_ID", "Patient_ID", "Recording_Time", "Position",
                   "Chunk_Index", "Equipment", "Acquisition_Mode",
                   "cycle_start", "cycle_end", "Label"]
    events_df = events_df[events_cols]
    events_path = OUT_DIR / "Events.csv"
    events_df.to_csv(events_path, index=False)

    print(f"\n  Events.csv -> {events_path}  ({len(events_df)} events)")
    print(f"\n  Label distribution:")
    print(events_df["Label"].value_counts().to_string())
    print(f"\n  Position distribution:")
    print(events_df["Position"].value_counts().head(10).to_string())
    print(f"\n  Equipment distribution:")
    print(events_df["Equipment"].value_counts().to_string())

    # ─────────────────────────────────────────────────────────────────────
    # Save Patient_info.csv
    # ─────────────────────────────────────────────────────────────────────
    steth_entries = []
    for key, pid in steth_registry._map.items():
        steth_entries.append({
            "Patient_ID":  pid,
            "Session_Key": key,
            "Source_Type":  "steth",
            "Equipment":   "Littmann 3200",
            "Acquisition_Mode": "sc",
            "Num_Recordings": int(
                events_df[events_df["Patient_ID"] == pid]["Recording_ID"].nunique()
            ),
        })

    trunc_entries = []
    for key, pid in trunc_registry._map.items():
        trunc_entries.append({
            "Patient_ID":  pid,
            "Session_Key": key,
            "Source_Type":  "trunc",
            "Equipment":   "HF-Type-1",
            "Acquisition_Mode": "mc",
            "Num_Recordings": int(
                events_df[events_df["Patient_ID"] == pid]["Recording_ID"].nunique()
            ),
        })

    patient_df = pd.DataFrame(steth_entries + trunc_entries)
    patient_path = OUT_DIR / "Patient_info.csv"
    patient_df.to_csv(patient_path, index=False)

    print(f"\n  Patient_info.csv -> {patient_path}")
    print(f"    Steth patients (S-IDs): {len(steth_registry)}")
    print(f"    Trunc patients (T-IDs): {len(trunc_registry)}")
    print(f"    Total patients:         {len(steth_registry) + len(trunc_registry)}")

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n  Audio files copied:  {n_copied}")
    print(f"  Audio files skipped: {n_skipped_copy} (already existed)")

    print("\n" + "=" * 62)
    print("  DONE!")
    print("=" * 62)
    print(f"  Output directory:  {OUT_DIR}")
    print(f"    Patient_info.csv : {len(patient_df)} patients")
    print(f"    Events.csv       : {len(events_df)} events")
    print(f"    Audio Files/     : {n_copied + n_skipped_copy} wav files")
    print()


if __name__ == "__main__":
    main()
