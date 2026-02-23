"""
3_data_preprocessing.py
───────────────────────
Reads IDX_data_filtered.xlsx (output of 2_data_preprocessing.py) and:
  • Encodes every 1-day percent-change column as binary:
        positive change (> 0) → 1 (UP)
        negative / zero change  → 0 (DOWN)
  • Renames columns from  *_CHG_PCT_1D  →  *_Direction
  • Prints a summary of UP/DOWN counts per index
  • Saves output: IDX_data_encoded.xlsx
"""

import pandas as pd
import os

input_fp = r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term 2\SMM748 Machine Learning Quantitative Finance\First Group Coursework ML\MS_Python_CW\input_data"

# ── 1. Load filtered data ──
df = pd.read_excel(os.path.join(input_fp, 'IDX_data_filtered.xlsx'))
df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce').dt.date
print(f'Loaded filtered data: {df.shape}')
print(f'Date range: {df["Dates"].min()} to {df["Dates"].max()}')
print(f'Columns: {list(df.columns)}\n')

# ── 2. Identify CHG_PCT_1D columns ──
chg_cols = [c for c in df.columns if 'CHG_PCT_1D' in c]
print(f'CHG_PCT_1D columns to encode ({len(chg_cols)}):')
for c in chg_cols:
    print(f'  {c}')

# ── 3. Encode: positive → 1 (UP), else → 0 (DOWN) ──
print(f'\n── Binary Encoding: positive change → 1 (UP), else → 0 (DOWN) ──')
summary = []

for col in chg_cols:
    # Encode
    df[col] = (df[col] > 0).astype(int)

    # Rename: NIFTY_CHG_PCT_1D → NIFTY_Direction
    new_name = col.replace('_CHG_PCT_1D', '_Direction')
    df.rename(columns={col: new_name}, inplace=True)

    # Summary stats
    up = df[new_name].sum()
    down = len(df) - up
    ratio = up / down if down > 0 else float('inf')
    summary.append({
        'Index': new_name.replace('_Direction', ''),
        'UP (1)': up,
        'DOWN (0)': down,
        'UP %': f'{up / len(df) * 100:.1f}%',
        'DOWN %': f'{down / len(df) * 100:.1f}%',
        'Imbalance (UP/DOWN)': f'{ratio:.3f}',
    })
    print(f'  {new_name:25s}  UP={up:3d} ({up/len(df)*100:.1f}%)  DOWN={down:3d} ({down/len(df)*100:.1f}%)  ratio={ratio:.3f}')

# ── 4. Print summary table ──
summary_df = pd.DataFrame(summary)
print(f'\n── Encoding Summary ──')
print(summary_df.to_string(index=False))

# ── 5. Drop VIX / volatility index columns ──
vix_prefixes = ['INVIXN', 'VXEFA', 'VXEEM', 'V2X', 'VHSI']
vix_cols = [c for c in df.columns if any(c.startswith(p) for p in vix_prefixes)]
df.drop(columns=vix_cols, inplace=True)
print(f'\nDropped {len(vix_cols)} VIX columns: {vix_cols}')

# ── 6. Preview encoded data ──
print(f'\nEncoded columns: {list(df.columns)}')
print(f'Final shape: {df.shape}')
print(df.head(10))

# ── 7. Save encoded file ──
output_path = os.path.join(input_fp, 'IDX_data_encoded.xlsx')
df.to_excel(output_path, index=False)
print(f'\nEncoded file saved to: {output_path}')
