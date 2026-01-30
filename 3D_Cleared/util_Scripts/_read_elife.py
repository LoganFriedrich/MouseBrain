#!/usr/bin/env python3
"""Temp script to read eLife data."""
import traceback

output_path = r"Y:\2_Connectome\3_Nuclei_Detection\util_Scripts\_elife_output.txt"

try:
    import pandas as pd
    xlsx_path = r"Y:\2_Connectome\Brainglobe\Comps\elife-76254-data1-v2.xlsx"

    # Read sheets
    key_df = pd.read_excel(xlsx_path, sheet_name='Key - labels')
    data_69 = pd.read_excel(xlsx_path, sheet_name='Data - 69 regions')
    data_25 = pd.read_excel(xlsx_path, sheet_name='Data - 25 grouped regions')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("KEY - Region Labels (69 to 25 mapping)\n")
        f.write("=" * 80 + "\n")
        f.write(key_df.to_string() + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("DATA - 69 Regions (first 20 rows, first 15 cols)\n")
        f.write("=" * 80 + "\n")
        f.write(data_69.iloc[:20, :15].to_string() + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("DATA - 25 Grouped Regions (all rows, first 15 cols)\n")
        f.write("=" * 80 + "\n")
        f.write(data_25.iloc[:, :15].to_string() + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("COLUMN HEADERS (Data - 69 regions)\n")
        f.write("=" * 80 + "\n")
        for i, col in enumerate(data_69.columns):
            f.write(f"  {i}: {col}\n")

except Exception as e:
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"ERROR: {e}\n\n")
        f.write(traceback.format_exc())
