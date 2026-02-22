"""
2_data_preprocessing.py
───────────────────────
Reads the preprocessed output from 1_data_preprocessing (IDX_data_preprocessed.xlsx)
and Trading_Holidays.xlsx to:
  • Remove rows where NIFTY (Indian market) was on holiday
  • Apply a start-date filter to keep < 500 rows
  • Log all actions
  • Save output: IDX_data_filtered.xlsx
"""

import pandas as pd
import os
import datetime

input_fp = r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term 2\SMM748 Machine Learning Quantitative Finance\First Group Coursework ML\MS_Python_CW\input_data"

# ── 1. Load preprocessed data (output of 1_data_preprocessing.py) ──
df = pd.read_excel(os.path.join(input_fp, 'IDX_data_preprocessed.xlsx'))
df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce').dt.date
print(f'Loaded preprocessed data: {df.shape}')
print(f'Date range: {df["Dates"].min()} to {df["Dates"].max()}')

# ── 2. Load Trading Holidays ──
holidays_file = os.path.join(input_fp, 'Trading_Holidays.xlsx')
xl = pd.ExcelFile(holidays_file)
print(f'\nTrading Holidays sheets: {xl.sheet_names}')

# Map each holiday sheet to the column prefixes it covers
HOLIDAY_INDEX_MAP = {
    'NIFTY_Holidays':  ['NIFTY', 'INVIXN'],
    'DJ_Holidays':     ['DJ'],
    'SP_Holidays':     ['SP'],
    'DAX_Holidays':    ['DAX', 'VXEFA', 'VXEEM', 'V2X'],
    'UKX_Holidays':    ['UKX'],
    'HSI_Holidays':    ['HSI', 'VHSI'],
    'SHCOMP_Holidays': ['SHCOMP'],
    'TWSE_Holidays':   ['TWSE'],
    'NKY_Holidays':    ['NKY'],
    'STI_Holidays':    ['STI'],
}

# Load all holiday date sets
holiday_data = {}
for sheet_name in xl.sheet_names:
    hdf = pd.read_excel(holidays_file, sheet_name=sheet_name)
    hdf['Date'] = pd.to_datetime(hdf['Date'], errors='coerce').dt.date
    holiday_data[sheet_name] = hdf
    print(f'  {sheet_name}: {len(hdf)} holidays loaded')

# ── 3. Remove rows where NIFTY was on holiday ──
nifty_hdf = holiday_data['NIFTY_Holidays']
nifty_holiday_dates = set(nifty_hdf['Date'])

rows_before = len(df)
holiday_mask = df['Dates'].isin(nifty_holiday_dates)
removed_rows = df[holiday_mask].copy()
df = df[~holiday_mask].reset_index(drop=True)
rows_after_holiday = len(df)
# ── 4. Date filter — keep < 500 rows ──
start_date = datetime.date(2024, 2, 14)
rows_before_date = len(df)
df = df[df['Dates'] >= start_date].reset_index(drop=True)
rows_after_date = len(df)
print(f'Date filter (>= {start_date}): {rows_before_date} -> {rows_after_date} rows ({rows_before_date - rows_after_date} removed)')

# ── 5. Create log ──
log_lines = []
log_lines.append('Data Filtering Log')
log_lines.append('=' * 60)
log_lines.append(f'Input file         : IDX_data_preprocessed.xlsx')
log_lines.append(f'Holidays file      : Trading_Holidays.xlsx')
log_lines.append(f'')

# NIFTY holiday removal
log_lines.append('-- NIFTY Holiday Filtering --')
log_lines.append(f'Total rows before  : {rows_before}')
log_lines.append(f'NIFTY holidays     : {len(nifty_holiday_dates)}')
log_lines.append(f'Rows removed       : {rows_before - rows_after_holiday}')
log_lines.append(f'Rows after         : {rows_after_holiday}')
log_lines.append(f'')

log_lines.append('Removed NIFTY holiday dates:')
for _, row in removed_rows.iterrows():
    match = nifty_hdf[nifty_hdf['Date'] == row['Dates']]
    name = match['Holiday'].values[0] if len(match) > 0 else 'Unknown'
    log_lines.append(f'  {row["Dates"]}  -  {name}')

log_lines.append(f'')
log_lines.append('NIFTY holidays with NO matching data row:')
matched = set(removed_rows['Dates'])
for d in sorted(nifty_holiday_dates):
    if d not in matched:
        match = nifty_hdf[nifty_hdf['Date'] == d]
        name = match['Holiday'].values[0] if len(match) > 0 else 'Unknown'
        log_lines.append(f'  {d}  -  {name}')

# Date filter
log_lines.append(f'')
log_lines.append('-- Date Filter --')
log_lines.append(f'Start date         : {start_date}')
log_lines.append(f'Rows removed       : {rows_before_date - rows_after_date}')
log_lines.append(f'')

# Final summary
log_lines.append('-- Final Summary --')
log_lines.append(f'Final row count    : {rows_after_date}')
log_lines.append(f'Final date range   : {df["Dates"].min()} to {df["Dates"].max()}')
log_lines.append(f'Columns            : {len(df.columns)}')
log_lines.append(f'')

# Per-exchange holiday coverage
log_lines.append('-- Holiday Coverage (all exchanges) --')
for sheet_name, prefixes in HOLIDAY_INDEX_MAP.items():
    hdf = holiday_data[sheet_name]
    h_dates = set(hdf['Date'])
    in_range = {d for d in h_dates if df['Dates'].min() <= d <= df['Dates'].max()}
    in_data = h_dates & set(df['Dates'])
    log_lines.append(f'  {sheet_name} ({", ".join(prefixes)}):')
    log_lines.append(f'    Total holidays: {len(hdf)}, In data range: {len(in_range)}, Still in filtered data: {len(in_data)}')

log_text = '\n'.join(log_lines)
print(f'\n{log_text}')

log_path = os.path.join(input_fp, 'holiday_filter_log.txt')
with open(log_path, 'w', encoding='utf-8') as f:
    f.write(log_text)
print(f'\nLog saved to: {log_path}')

# ── 6. Save filtered file ──
print(f'\nFinal shape: {df.shape}')
print(df.head())

output_path = os.path.join(input_fp, 'IDX_data_filtered.xlsx')
df.to_excel(output_path, index=False)
print(f'\nFiltered file saved to: {output_path}')
