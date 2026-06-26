"""
=============================================================================
  Lung Sound Dataset Merge
  Datasets: ICBHI 2017 | HF-Lung V1 | SPR
  Target:   Crackle / Wheeze / Normal detection
=============================================================================

Output structure:
  merged_dataset/
    manifest.csv          <- metadata for every cycle (with location)
    locations/            <- separate CSVs for each standard location
      manifest_anterior_left.csv
      ...
    audio/                <- all unique source wav files copied here
"""

import re
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE    = Path(r"c:\Spectrolung")
OUT_DIR = BASE / "merged_dataset"
AUD_DIR = OUT_DIR / "audio"
CSV_OUT = OUT_DIR / "manifest.csv"
LOC_DIR = OUT_DIR / "locations"

# Create output folders
OUT_DIR.mkdir(exist_ok=True)
AUD_DIR.mkdir(exist_ok=True)
LOC_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: register a source wav -> unique destination filename
# ─────────────────────────────────────────────────────────────────────────────
_copy_registry: dict[Path, str] = {}

def _register_audio(prefix: str, src_path: Path) -> str:
    """Return the destination filename for src_path (deduplicated)."""
    if src_path in _copy_registry:
        return _copy_registry[src_path]
    dest_name = f"{prefix}_{src_path.name}"
    _copy_registry[src_path] = dest_name
    return dest_name


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ICBHI 2017
# ─────────────────────────────────────────────────────────────────────────────
def parse_icbhi() -> pd.DataFrame:
    audio_dir = (BASE / "Respiratory_Sound_Database"
                      / "Respiratory_Sound_Database"
                      / "audio_and_txt_files")
    records = []
    skipped = 0

    txt_files = sorted(audio_dir.glob("*.txt"))
    print(f"[ICBHI] Found {len(txt_files)} annotation files")
    
    loc_map = {
        'Tc': 'trachea',
        'Al': 'anterior_left', 'Ar': 'anterior_right',
        'Pl': 'posterior_left', 'Pr': 'posterior_right',
        'Ll': 'lateral_left', 'Lr': 'lateral_right'
    }

    for txt_file in txt_files:
        wav_file = txt_file.with_suffix(".wav")
        if not wav_file.exists():
            skipped += 1
            continue

        dest_name = _register_audio("icbhi", wav_file)
        
        parts = wav_file.stem.split('_')
        loc_raw = parts[2] if len(parts) >= 3 else ""
        location = loc_map.get(loc_raw, 'unknown')

        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                try:
                    start_s = float(parts[0])
                    end_s   = float(parts[1])
                    crackle = int(float(parts[2]))
                    wheeze  = int(float(parts[3]))
                except ValueError:
                    continue

                if crackle == 1 and wheeze == 0:
                    label, label_fine = "crackle", "crackle"
                elif crackle == 0 and wheeze == 1:
                    label, label_fine = "wheeze",  "wheeze"
                elif crackle == 1 and wheeze == 1:
                    label, label_fine = "crackle", "crackle_wheeze"
                else:
                    label, label_fine = "normal",  "normal"

                records.append(dict(
                    audio_filename=dest_name,
                    start_s=start_s,
                    end_s=end_s,
                    label=label,
                    label_fine=label_fine,
                    source="icbhi",
                    location=location,
                ))

    df = pd.DataFrame(records)
    print(f"[ICBHI] Parsed {len(df)} cycles  |  skipped {skipped} missing wavs")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HF-Lung V1
# ─────────────────────────────────────────────────────────────────────────────
_TS_RE = re.compile(r"(\d+):(\d+):([\d.]+)")

def _ts_to_s(ts: str) -> float:
    m = _TS_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Cannot parse timestamp: {ts!r}")
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s

def _overlaps(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1

def parse_hf_lung() -> pd.DataFrame:
    EXCLUDED_LABELS = {"Rhonchi", "Stridor"}
    records    = []
    n_excluded = 0
    n_no_wav   = 0
    n_files    = 0

    loc_map = {
        'L1': 'anterior_right', 'L2': 'anterior_right',
        'L3': 'lateral_right', 'L4': 'lateral_right',
        'L5': 'anterior_left', 'L6': 'anterior_left',
        'L7': 'lateral_left', 'L8': 'lateral_left'
    }

    for split in ["train", "test"]:
        split_dir   = BASE / "HF_lung_V1" / split
        label_files = sorted(split_dir.glob("*_label.txt"))
        print(f"[HF-Lung] Processing {split}: {len(label_files)} label files ...")

        for label_file in label_files:
            stem     = label_file.stem.replace("_label", "")
            wav_file = split_dir / f"{stem}.wav"
            if not wav_file.exists():
                n_no_wav += 1
                continue

            n_files   += 1
            dest_name  = _register_audio("hf_lung", wav_file)
            
            m = re.search(r'-(L\d)_', stem)
            loc_raw = m.group(1) if m else ""
            location = loc_map.get(loc_raw, 'unknown')

            # Parse annotation events
            inspirations = []
            expirations  = []
            wheezes      = []
            crackles     = [] # D labels
            excluded_ev  = []

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
                    elif ev_label == "Wheeze":
                        wheezes.append((t0, t1))
                    elif ev_label == "D":
                        crackles.append((t0, t1))
                    elif ev_label in EXCLUDED_LABELS:
                        excluded_ev.append((t0, t1))

            if not inspirations:
                continue

            inspirations.sort(key=lambda x: x[0])
            expirations.sort(key=lambda x: x[0])

            # Pair each I with its following E
            cycles  = []
            exp_idx = 0
            for i_start, i_end in inspirations:
                cycle_end = i_end
                for j in range(exp_idx, len(expirations)):
                    e_start, e_end = expirations[j]
                    if e_start >= i_end - 0.15:
                        cycle_end = e_end
                        exp_idx   = j + 1
                        break
                cycles.append((i_start, cycle_end))

            # Label each cycle
            for c_start, c_end in cycles:
                if any(_overlaps(c_start, c_end, xs, xe) for xs, xe in excluded_ev):
                    n_excluded += 1
                    continue

                wheeze_hit = any(_overlaps(c_start, c_end, ws, we) for ws, we in wheezes)
                crackle_hit = any(_overlaps(c_start, c_end, cs, ce) for cs, ce in crackles)
                
                if crackle_hit and wheeze_hit:
                    label, label_fine = "crackle", "crackle_wheeze"
                elif crackle_hit:
                    label, label_fine = "crackle", "crackle"
                elif wheeze_hit:
                    label, label_fine = "wheeze", "wheeze"
                else:
                    label, label_fine = "normal", "normal"

                records.append(dict(
                    audio_filename=dest_name,
                    start_s=c_start,
                    end_s=c_end,
                    label=label,
                    label_fine=label_fine,
                    source="hf_lung",
                    location=location,
                ))

    df = pd.DataFrame(records)
    print(f"[HF-Lung] Processed {n_files} files  |  dropped {n_excluded} cycles (Rhonchi/Stridor overlap)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SPR
# ─────────────────────────────────────────────────────────────────────────────
_SPR_LABEL_MAP = {
    "Normal":          ("normal",  "normal"),
    "Fine Crackle":    ("crackle", "fine_crackle"),
    "Coarse Crackle":  ("crackle", "coarse_crackle"),
    "Wheeze":          ("wheeze",  "wheeze"),
    "Wheeze+Crackle":  ("crackle", "crackle_wheeze"),
}
_SPR_EXCLUDED = {"Rhonchi", "Stridor"}

def parse_spr() -> pd.DataFrame:
    events_csv = BASE / "SPR" / "events.csv"
    records_csv = BASE / "SPR" / "records.csv"
    audio_dir  = BASE / "SPR" / "audio_files"

    events   = pd.read_csv(events_csv)
    
    # We need location from records.csv which maps recording_id -> chest_location
    rec_df = pd.read_csv(records_csv)
    # create a fast dict mapping recording_id -> chest_location
    loc_dict = dict(zip(rec_df['recording_id'].astype(str), rec_df['chest_location'].astype(str)))
    
    loc_map = {
        'p1': 'posterior_left', 'p2': 'lateral_left',
        'p3': 'posterior_right', 'p4': 'lateral_right'
    }

    records  = []
    n_excl   = 0
    n_unk    = 0
    n_no_wav = 0

    print(f"[SPR] Found {len(events)} event rows")

    for _, row in events.iterrows():
        cl = str(row["clinical_label"]).strip()

        if cl in _SPR_EXCLUDED:
            n_excl += 1
            continue
        if cl not in _SPR_LABEL_MAP:
            n_unk += 1
            continue

        label, label_fine = _SPR_LABEL_MAP[cl]
        recording_id      = str(row["recording_id"]).strip()
        wav_file          = audio_dir / f"{recording_id}.wav"

        if not wav_file.exists():
            n_no_wav += 1
            continue
            
        loc_raw = loc_dict.get(recording_id, "").strip()
        location = loc_map.get(loc_raw, "unknown")

        dest_name = _register_audio("spr", wav_file)

        records.append(dict(
            audio_filename=dest_name,
            start_s=float(row["start_ms"]) / 1000.0,
            end_s=float(row["end_ms"])     / 1000.0,
            label=label,
            label_fine=label_fine,
            source="spr",
            location=location,
        ))

    df = pd.DataFrame(records)
    print(f"[SPR] Parsed {len(df)} cycles")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Copy audio files to merged_dataset/audio/
# ─────────────────────────────────────────────────────────────────────────────
def copy_audio_files() -> None:
    print(f"Copying {len(_copy_registry)} unique audio files -> {AUD_DIR} ...")
    for src_path, dest_name in tqdm(_copy_registry.items(), unit="file"):
        dest_path = AUD_DIR / dest_name
        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)
    print("Copy complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Validate + Save manifests
# ─────────────────────────────────────────────────────────────────────────────
def validate_and_save(df: pd.DataFrame) -> None:
    print("=" * 62)
    print("  VALIDATION")
    print("=" * 62)

    # No NaN in key columns
    key_cols = ["audio_filename", "start_s", "end_s",
                "label", "label_fine", "source", "location"]
    for col in key_cols:
        n_null = df[col].isna().sum()
        print(f"  {col:20s}  {'OK' if n_null == 0 else f'FAIL  {n_null} NaN!'}")

    print()
    print("=" * 62)
    print("  LABEL DISTRIBUTION")
    print("=" * 62)
    overall = df.groupby("label").size().reset_index(name="count")
    overall["pct"] = (overall["count"] / len(df) * 100).round(1)
    print(overall.to_string(index=False))
    
    print("\n  -- By Location --")
    locs = df.groupby("location").size().reset_index(name="count")
    print(locs.to_string(index=False))
    
    print(f"\n  Total cycles : {len(df):,}")

    # Save Main CSV
    df = df[["audio_filename", "start_s", "end_s",
             "label", "label_fine", "source", "location"]]
    df.to_csv(CSV_OUT, index=False)
    
    # Save Split CSVs
    for loc in df['location'].unique():
        sub_df = df[df['location'] == loc]
        sub_df.to_csv(LOC_DIR / f"manifest_{loc}.csv", index=False)

    print(f"\n  manifest.csv -> {CSV_OUT}")
    print(f"  locations/   -> {LOC_DIR}")
    print(f"  audio/       -> {AUD_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  PARSING DATASETS")
    print("=" * 62 + "\n")

    df_icbhi   = parse_icbhi()
    df_hf_lung = parse_hf_lung()
    df_spr     = parse_spr()

    df = pd.concat([df_icbhi, df_hf_lung, df_spr], ignore_index=True)

    # Copy all unique source wavs into merged_dataset/audio/
    copy_audio_files()

    validate_and_save(df)
