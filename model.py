import csv
import urllib.parse
import uuid
from collections import defaultdict
from label_studio_ml.model import LabelStudioMLBase
import re

# --- Strict label mapping -----------------------------------------------
LABEL_MAP = {
    "inhalation": "Inhalation",
    "exhalation": "Exhalation",
    "coarse_crackle": "Coarse Crackle",
    "rhonchi": "Rhonchi",
    "wheeze": "Wheeze",
    "stridor": "Stridor",
    "normal_breath_sound": "Normal",
    "artifact_possible": "Artifact Possible",
    "needs_audio_review": "Needs Audio Review",
    "fine_crackle": "Fine Crackle",
    "pleural_rub": "Pleural Rub"
}

CSV_FILE = "C:\\Spectrolung\\hf_lung_relabel_windows_v1.csv"

def sanitize_filename(filename):
    """
    Strips the Label Studio hash and removes ALL delimiters (spaces, hyphens, underscores).
    This guarantees a match no matter how the OS or Label Studio sanitized the string.
    
    CSV: 'steth_20180814_10_56_19 ak 26.wav'    -> 'steth20180814105619ak26'
    LS:  '8f3a9c-steth_20180814_10_56_19_ak_26' -> 'steth20180814105619ak26'
    
    CSV: 'steth_20180814_09_38_28 - ak1.wav'   -> 'steth20180814093828ak1'
    LS:  '2da8ed6c-steth_20180814_09_38_28_ak1' -> 'steth20180814093828ak1'
    """
    # 1. Strip the Label Studio prefix hash if it exists
    if "-" in filename:
        # Check if the part before the hyphen looks like a short hash
        prefix = filename.split("-", 1)[0]
        if len(prefix) <= 10:
            filename = filename.split("-", 1)[1]
            
    # 2. Remove the extension and make lowercase
    clean = filename.lower().replace(".wav", "")
    
    # 3. The Nuclear Option: Strip all non-alphanumeric characters.
    # This removes all spaces ' ', hyphens '-', and underscores '_'
    clean = re.sub(r'[^a-z0-9]', '', clean)
        
    return clean

class CSVLookupModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(CSVLookupModel, self).__init__(**kwargs)
        
        self.annotations = defaultdict(list)
        try:
            with open(CSV_FILE, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row["label"].strip()
                    if label in LABEL_MAP:
                        # Group using the sanitized filename
                        clean_key = sanitize_filename(row["filename"].strip())
                        self.annotations[clean_key].append(row)
            print(f"✅ ML Backend Ready: Loaded annotations for {len(self.annotations)} unique files.")
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")

    def predict(self, tasks, **kwargs):
        predictions = []

        for task in tasks:
            audio_url = task.get("data", {}).get("audio", "")
            decoded_url = urllib.parse.unquote(audio_url)
            
            # Extract just the filename at the end of the URL path
            raw_filename = decoded_url.split("/")[-1]
            
            # Sanitize the incoming filename
            search_key = sanitize_filename(raw_filename)

            print(f"\n🔍 Processing Task ID: {task.get('id')}")
            print(f"   -> Raw LS URL: '{raw_filename}'")
            print(f"   -> Search Key: '{search_key}'")

            results = []
            
            # Direct dictionary lookup - O(1) matching
            if search_key in self.annotations:
                print(f"   ✅ Exact match found! Building predictions...")
                grouped_regions = {}
                
                # Group overlapping labels and metadata
                for row in self.annotations[search_key]:
                    start = float(row["start_s"])
                    end = float(row["end_s"])
                    label = LABEL_MAP[row["label"].strip()]
                    
                    if (start, end) not in grouped_regions:
                        grouped_regions[(start, end)] = {'labels': set(), 'metadata': []}
                        
                    grouped_regions[(start, end)]['labels'].add(label)
                    
                    # Format metadata
                    conf = row.get("confidence", "").strip()
                    src = row.get("source", "").strip()
                    notes = row.get("notes", "").strip()
                    
                    meta_string = f"[{label}] Source: {src} | Conf: {conf}"
                    if notes:
                        meta_string += f"\nNote: {notes}"
                        
                    if meta_string not in grouped_regions[(start, end)]['metadata']:
                        grouped_regions[(start, end)]['metadata'].append(meta_string)

                # Convert grouped dictionaries into Label Studio JSON format
                for (start, end), data in grouped_regions.items():
                    region_id = str(uuid.uuid4())[:8]
                    
                    results.append({
                        "id": region_id,
                        "type": "labels",
                        "from_name": "labels",
                        "to_name": "audio",
                        "value": {
                            "start": start,
                            "end": end,
                            "labels": list(data['labels']),
                        }
                    })
                    
                    if data['metadata']:
                        combined_notes = "\n\n".join(data['metadata'])
                        results.append({
                            "id": region_id,
                            "type": "textarea",
                            "from_name": "notes",
                            "to_name": "audio",
                            "value": {
                                "start": start,
                                "end": end,
                                "text": [combined_notes],
                            }
                        })
            else:
                print(f"   ❌ No match found in CSV for key '{search_key}'.")

            predictions.append({
                "model_version": "csv_lookup_exact_match",
                "result": results
            })

        return predictions