"""
1_data_preprocessing.py
───────────────────────
Reads the raw Bloomberg IDX_data_v1.xlsx (Sheet1):
  • Skips the first 6 metadata/header rows
  • Dynamically detects index names (row 4) and field names (row 6)
  • Renames columns with friendly prefixes:
        NIFTY Index   → NIFTY_
        INDU Index    → DJ_
        SPX Index     → SP_
        DAX Index     → DAX_
        UKX Index     → UKX_
        HSI Index     → HSI_
        SHCOMP Index  → SHCOMP_
        TWSE Index    → TWSE_
        NKY Index     → NKY_
        INVIXN Index  → INVIXN_
        VXEFA Index   → VXEFA_
        VXEEM Index   → VXEEM_
        V2X Index     → V2X_
        STI Index     → STI_
        VHSI Index    → VHSI_
  • Converts Dates to date-only format
  • Saves intermediate output: IDX_data_preprocessed.xlsx
"""

import pandas as pd
import os

input_fp = r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term 2\SMM748 Machine Learning Quantitative Finance\First Group Coursework ML\MS_Python_CW\input_data"

# ── Friendly prefix mapping ──
PREFIX_MAP = {
    'NIFTY':  'NIFTY',
    'INDU':   'DJ',
    'SPX':    'SP',
    'DAX':    'DAX',
    'UKX':    'UKX',
    'HSI':    'HSI',
    'SHCOMP': 'SHCOMP',
    'TWSE':   'TWSE',
    'NKY':    'NKY',
    'INVIXN': 'INVIXN',
    'VXEFA':  'VXEFA',
    'VXEEM':  'VXEEM',
    'V2X':    'V2X',
    'STI':    'STI',
    'VHSI':   'VHSI',
}

# ── 1. Read raw file (no header) ──
raw = pd.read_excel(os.path.join(input_fp, 'IDX_data_v1.xlsx'),
                     sheet_name='Sheet1', header=None)
print(f'Raw data shape: {raw.shape}')

# ── 2. Extract index names (row 3) and field names (row 5) ──
index_row = raw.iloc[3]
field_row  = raw.iloc[5]

# Collect raw short names in order
raw_index_names = []
for val in index_row:
    if pd.notna(val):
        short = str(val).split()[0]           # "NIFTY Index" → "NIFTY"
        raw_index_names.append(short)

# Collect field labels from row 5 (skip col 0 = "Dates")
field_labels = []
for col_idx in range(1, raw.shape[1]):
    val = field_row.iloc[col_idx]
    field_labels.append(str(val) if pd.notna(val) else f'FIELD_{col_idx}')

# Work out how many fields per index
fields_per_index = len(field_labels) // len(raw_index_names)
print(f'Detected {len(raw_index_names)} indices, {fields_per_index} fields each')
print(f'Indices: {raw_index_names}')

# ── 3. Build new column names with friendly prefixes ──
new_columns = ['Dates']
for i, raw_name in enumerate(raw_index_names):
    prefix = PREFIX_MAP.get(raw_name, raw_name)
    start = i * fields_per_index
    for j in range(fields_per_index):
        new_columns.append(f'{prefix}_{field_labels[start + j]}')

print(f'New columns ({len(new_columns)}): {new_columns}')

# ── 4. Build the DataFrame — data starts at row 6 (0-indexed) ──
df = raw.iloc[6:].copy()
df.columns = new_columns
df.reset_index(drop=True, inplace=True)

# Convert Dates to date-only
df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce').dt.date

print(f'\nPreprocessed shape: {df.shape}')
print(df.head())

# ── 5. Save preprocessed output ──
output_path = os.path.join(input_fp, 'IDX_data_preprocessed.xlsx')
df.to_excel(output_path, index=False)
print(f'\nPreprocessed file saved to: {output_path}')
