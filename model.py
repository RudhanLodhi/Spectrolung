import csv
import urllib.parse
import uuid
from collections import defaultdict
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_env

# --- label mapping -------------------------------------------------------
LABEL_MAP = {
    "I": "Inhale",
    "E": "Exhale",
    "D": "Coarse Crackle",
    "Rhonchi": "Rhonchi",
    "Wheeze": "Wheeze",
    "Stridor": "Stridor",
}

CSV_FILE = "C:\\Spectrolung\\RAW_Datasets\\HF_lung_V1_RAW\\annotations.csv" # Update path if needed

class CSVLookupModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(CSVLookupModel, self).__init__(**kwargs)
        
        # Load the CSV into memory once when the server starts
        self.annotations = defaultdict(list)
        try:
            with open(CSV_FILE, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row["label"].strip()
                    if label in LABEL_MAP:
                        self.annotations[row["filename"].strip()].append(row)
            print(f"✅ ML Backend Ready: Loaded annotations for {len(self.annotations)} files.")
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")

    def predict(self, tasks, **kwargs):
        """This function is called every time Label Studio requests a prediction."""
        predictions = []

        for task in tasks:
            # Get the internal URL created by Label Studio
            audio_url = task.get("data", {}).get("audio", "")
            decoded_url = urllib.parse.unquote(audio_url)
            
            results = []
            matched_filename = None
            
            # Find which of our original CSV filenames is inside this URL
            for original_filename in self.annotations.keys():
                if original_filename in decoded_url:
                    matched_filename = original_filename
                    break

            # If we found a match, build the predictions format
            if matched_filename:
                for row in self.annotations[matched_filename]:
                    results.append({
                        "id": str(uuid.uuid4())[:8],
                        "type": "labels",
                        "from_name": "labels", # Matches the Name attribute in your XML
                        "to_name": "audio",    # Matches the toName attribute in your XML
                        "value": {
                            "start": float(row["start_time"]),
                            "end": float(row["end_time"]),
                            "labels": [LABEL_MAP[row["label"].strip()]],
                        }
                    })

            # Append the result for this task
            predictions.append({
                "model_version": "csv_lookup_v1",
                "result": results
            })

        return predictions