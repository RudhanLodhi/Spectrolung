import pandas as pd
import os

def update_chest_locations():
    # Target the raw CSV directly
    csv_path = "../SPR/records.csv"
    
    # Define the ICBHI mapping
    position_mapping = {
        'p1': 'Al', 'P1': 'Al',
        'p2': 'Ar', 'P2': 'Ar',
        'p3': 'Pl', 'P3': 'Pl',
        'p4': 'Pr', 'P4': 'Pr'
    }

    # Check if the file exists
    if not os.path.exists(csv_path):
        # Fallback to current directory just in case the script is run from the root
        csv_path = "./SPR/records.csv"
        if not os.path.exists(csv_path):
            print(f"Error: Could not find records.csv at {csv_path}")
            return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # The SPR dataset sometimes calls this column 'loc' or 'chest_location'. 
    # We will check for 'chest_location' first, then fallback to 'loc'.
    target_col = 'chest_location'
    if target_col not in df.columns:
        if 'loc' in df.columns:
            target_col = 'loc'
        else:
            print(f"Error: Neither 'chest_location' nor 'loc' found in columns: {list(df.columns)}")
            return
            
    # Apply the mapping. If a value isn't p1-p4, it keeps its original value.
    df[target_col] = df[target_col].astype(str).map(lambda x: position_mapping.get(x.strip(), x))
    
    # Save it back to the exact same location, overwriting the old one
    df.to_csv(csv_path, index=False)
    print(f"✅ Successfully updated '{target_col}' values and saved back to {csv_path}!")

if __name__ == "__main__":
    update_chest_locations()