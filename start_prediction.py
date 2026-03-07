import pandas as pd
from pathlib import Path

import torch
if torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)
from prediiction import process_case

def process_excel_files(folder_path):
    directory = Path(folder_path)
    excel_files = sorted(directory.glob("*.xlsx"))
    
    if not excel_files:
        print("No Excel files found in the directory.")
        return

    print(f"Found {len(excel_files)} files. Starting processing...")
    metrics_summary = []

    for file_path in excel_files:
        print(f"Processing: {file_path.name}")
        try:
            data = process_case(file_path)
            metrics_summary.append((file_path.name, data))
            print(f"   Successfully processed {file_path.name}.")
        except Exception as e:
            print(f"   Error processing {file_path.name}: {e}")
    print()
    print()
    print()
    print("\nAll files processed. Summary of metrics:")
    for filename, m in metrics_summary:
        print(filename)
        print(m)
        print()
        print()

folder_to_scan = 'data/manual_clean' 
process_excel_files(folder_to_scan)

