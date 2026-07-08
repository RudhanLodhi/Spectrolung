"""
=============================================================================
  ICBHI 2017 Dataset Preprocessing Script
  Restructures the raw Respiratory_Sound_Database into a standardized format.
=============================================================================

Output structure:
  ICBHI/
    Audio Files/        <- all .wav files copied over
    Patient_info.csv    <- merged demographic + diagnosis info
    Events.csv          <- parsed annotation cycles with inferred labels
"""

import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE      = Path(r"C:\Spectrolung")
RAW_DIR   = BASE / "Respiratory_Sound_Database"
INNER_DIR = RAW_DIR / "Respiratory_Sound_Database"
AUDIO_SRC = INNER_DIR / "audio_and_txt_files"

OUT_DIR   = BASE / "ICBHI"
AUD_OUT   = OUT_DIR / "Audio Files"

# Create output directories
OUT_DIR.mkdir(exist_ok=True)
AUD_OUT.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build Patient_info.csv
#    Merge demographic_info.txt + patient_diagnosis.csv
# ─────────────────────────────────────────────────────────────────────────────
def build_patient_info() -> pd.DataFrame:
    print("=" * 62)
    print("  BUILDING Patient_info.csv")
    print("=" * 62)

    # --- Parse demographic_info.txt ---
    # Format: PatientID  Age  Sex  BMI  Weight  Height
    # (space-separated, "NA" for missing values)
    demo_path = RAW_DIR / "demographic_info.txt"
    demo_rows = []
    with open(demo_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            pid, age, sex, bmi, weight, height = parts[:6]
            demo_rows.append({
                "Patient_ID": int(pid),
                "Age":        float(age) if age != "NA" else None,
                "Sex":        sex if sex != "NA" else None,
                "BMI":        float(bmi) if bmi != "NA" else None,
                "Weight":     float(weight) if weight != "NA" else None,
            })
    demo_df = pd.DataFrame(demo_rows)
    print(f"  Parsed {len(demo_df)} patients from demographic_info.txt")

    # --- Parse patient_diagnosis.csv ---
    # Format: PatientID,Diagnosis (no header)
    diag_path = INNER_DIR / "patient_diagnosis.csv"
    diag_df = pd.read_csv(diag_path, header=None, names=["Patient_ID", "Diagnosis"])
    print(f"  Parsed {len(diag_df)} patients from patient_diagnosis.csv")

    # --- Merge ---
    merged = pd.merge(demo_df, diag_df, on="Patient_ID", how="outer")
    merged = merged.sort_values("Patient_ID").reset_index(drop=True)

    # Reorder columns
    merged = merged[["Patient_ID", "Age", "Sex", "BMI", "Weight", "Diagnosis"]]

    out_path = OUT_DIR / "Patient_info.csv"
    merged.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}  ({len(merged)} patients)")
    print(f"\n  Diagnosis distribution:")
    print(merged["Diagnosis"].value_counts().to_string())
    print()

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build Events.csv
#    Parse .txt annotation files, extract metadata from filenames,
#    infer crackle type from diagnosis
# ─────────────────────────────────────────────────────────────────────────────

# Crackle-type inference based on patient diagnosis
_FINE_CRACKLE_DIAG    = {"Pneumonia", "Bronchiolitis"}
_COARSE_CRACKLE_DIAG  = {"COPD", "Bronchiectasis", "LRTI", "URTI"}
# Healthy, Asthma, or anything else defaults to Coarse Crackle


def _infer_crackle_type(diagnosis: str) -> str:
    """Infer Fine vs Coarse crackle from patient diagnosis."""
    if diagnosis in _FINE_CRACKLE_DIAG:
        return "Fine Crackle"
    else:
        # COPD, Bronchiectasis, LRTI, URTI, Healthy, Asthma, unknown
        return "Coarse Crackle"


def build_events(patient_df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 62)
    print("  BUILDING Events.csv")
    print("=" * 62)

    # Build a quick lookup: patient_id -> diagnosis
    diag_lookup = dict(zip(
        patient_df["Patient_ID"].astype(int),
        patient_df["Diagnosis"].fillna("Unknown")
    ))

    txt_files = sorted(AUDIO_SRC.glob("*.txt"))
    print(f"  Found {len(txt_files)} annotation files\n")

    records = []
    n_skipped = 0

    for txt_file in tqdm(txt_files, desc="  Parsing annotations", unit="file"):
        wav_file = txt_file.with_suffix(".wav")
        if not wav_file.exists():
            n_skipped += 1
            continue

        # --- Parse filename ---
        # Format: PatientID_RecordingIndex_Position_AcqMode_Equipment
        # Example: 101_1b1_Al_sc_Meditron
        stem = wav_file.stem
        parts = stem.split("_")

        if len(parts) >= 5:
            patient_id = int(parts[0])
            position   = parts[2]              # e.g. Al, Pr, Tc, Ll
            acq_mode   = parts[3]              # sc or mc
            equipment  = "_".join(parts[4:])   # e.g. Meditron, Litt3200, AKGC417L
        elif len(parts) >= 4:
            patient_id = int(parts[0])
            position   = parts[2]
            acq_mode   = parts[3]
            equipment  = "Unknown"
        else:
            patient_id = int(parts[0]) if parts[0].isdigit() else 0
            position   = parts[2] if len(parts) > 2 else "Unknown"
            acq_mode   = parts[3] if len(parts) > 3 else "Unknown"
            equipment  = "Unknown"

        recording_id = stem
        diagnosis = diag_lookup.get(patient_id, "Unknown")

        # --- Parse annotation lines ---
        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split("\t")
                if len(cols) < 4:
                    continue
                try:
                    cycle_start = float(cols[0])
                    cycle_end   = float(cols[1])
                    crackle_flag = int(float(cols[2]))
                    wheeze_flag  = int(float(cols[3]))
                except ValueError:
                    continue

                # --- Labeling logic ---
                if crackle_flag == 0 and wheeze_flag == 0:
                    label = "Normal"
                elif crackle_flag == 0 and wheeze_flag == 1:
                    label = "Wheeze"
                elif crackle_flag == 1 and wheeze_flag == 1:
                    label = "Wheeze+Crackle"
                elif crackle_flag == 1 and wheeze_flag == 0:
                    # Infer crackle type from patient diagnosis
                    label = _infer_crackle_type(diagnosis)
                else:
                    label = "Normal"

                records.append({
                    "Recording_ID":    recording_id,
                    "Position":        position,
                    "Acquisition_Mode": acq_mode,
                    "Equipment":       equipment,
                    "cycle_start":     cycle_start,
                    "cycle_end":       cycle_end,
                    "Label":           label,
                })

    df = pd.DataFrame(records)
    out_path = OUT_DIR / "Events.csv"
    df.to_csv(out_path, index=False)

    print(f"\n  Saved -> {out_path}  ({len(df)} events)")
    print(f"  Skipped {n_skipped} annotation files (missing .wav)")
    print(f"\n  Label distribution:")
    print(df["Label"].value_counts().to_string())
    print(f"\n  Position distribution:")
    print(df["Position"].value_counts().to_string())
    print(f"\n  Equipment distribution:")
    print(df["Equipment"].value_counts().to_string())
    print(f"\n  Acquisition Mode distribution:")
    print(df["Acquisition_Mode"].value_counts().to_string())
    print()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Copy audio files
# ─────────────────────────────────────────────────────────────────────────────
def copy_audio_files():
    print("=" * 62)
    print("  COPYING AUDIO FILES")
    print("=" * 62)

    wav_files = sorted(AUDIO_SRC.glob("*.wav"))
    print(f"  Found {len(wav_files)} .wav files\n")

    copied = 0
    skipped = 0
    for wav in tqdm(wav_files, desc="  Copying", unit="file"):
        dest = AUD_OUT / wav.name
        if not dest.exists():
            shutil.copy2(wav, dest)
            copied += 1
        else:
            skipped += 1

    print(f"\n  Copied {copied} new files, skipped {skipped} (already exist)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  ICBHI 2017 PREPROCESSING")
    print("=" * 62 + "\n")

    patient_df = build_patient_info()
    events_df  = build_events(patient_df)
    copy_audio_files()

    print("=" * 62)
    print("  DONE!")
    print("=" * 62)
    print(f"  Output directory: {OUT_DIR}")
    print(f"    Patient_info.csv : {len(patient_df)} patients")
    print(f"    Events.csv       : {len(events_df)} events")
    print(f"    Audio Files/     : {len(list(AUD_OUT.glob('*.wav')))} wav files")
    print()
