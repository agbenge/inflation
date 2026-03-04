import pandas as pd
from pathlib import Path

from prediiction import process_case

def process_excel_files(folder_path):
    directory = Path(folder_path)
    excel_files = list(directory.glob("*.xlsx"))
    
    if not excel_files:
        print("No Excel files found in the directory.")
        return

    print(f"Found {len(excel_files)} files. Starting processing...")

    for file_path in excel_files:
        print(f"Processing: {file_path.name}")
        # try:
        process_case(file_path)
        print(f"   Successfully processed {file_path.name}.")
            
        # except Exception as e:
        #     print(f"   Error processing {file_path.name}: {e}")

folder_to_scan = 'data/manual_clean' 
process_excel_files(folder_to_scan)