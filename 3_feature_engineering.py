"""
3_feature_engineering.py
────────────────────────
Reads IDX_data_filtered.xlsx and:
  1. Creates the binary target: NIFTY_Direction  (1 = UP, 0 = DOWN)
  2. Builds lagged features (lag 1–10) for all CHG_PCT columns
  3. Computes the imbalance ratio of the target
  4. Runs kNN across different lag windows to find which lags
     best predict NIFTY direction
  5. Saves the final modelling-ready dataset: IDX_model_data.xlsx
  6. Saves the kNN lag analysis results: knn_lag_analysis.xlsx
"""

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

input_fp = r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term 2\SMM748 Machine Learning Quantitative Finance\First Group Coursework ML\MS_Python_CW\input_data"

# ── 1. Load filtered data ──
df = pd.read_excel(os.path.join(input_fp, 'IDX_data_filtered.xlsx'))
df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
print(f'Loaded filtered data: {df.shape}')

# ── 2. Create binary target ──
# NIFTY_CHG_PCT_1D > 0  →  1 (UP),  else  0 (DOWN)
df['NIFTY_Direction'] = (df['NIFTY_CHG_PCT_1D'] > 0).astype(int)

up_count = df['NIFTY_Direction'].sum()
down_count = len(df) - up_count
imbalance_ratio = up_count / down_count if down_count > 0 else float('inf')

print(f'\n── Imbalance Analysis ──')
print(f'UP   (1): {up_count}  ({up_count/len(df)*100:.1f}%)')
print(f'DOWN (0): {down_count}  ({down_count/len(df)*100:.1f}%)')
print(f'Imbalance ratio (UP/DOWN): {imbalance_ratio:.3f}')

# ── 3. Identify feature columns for lagging ──
# Use CHG_PCT columns from ALL indices (including NIFTY lagged values)
chg_cols = [c for c in df.columns if 'CHG_PCT' in c]
print(f'\nCHG_PCT columns for lag features ({len(chg_cols)}): {chg_cols}')

# ── 4. Create lagged features (lag 1 to max_lag) ──
max_lag = 10

print(f'\nCreating lagged features (lag 1 to {max_lag})...')
lag_columns = {}  # lag_number -> list of column names
for lag in range(1, max_lag + 1):
    lag_cols = []
    for col in chg_cols:
        lag_col_name = f'{col}_lag{lag}'
        df[lag_col_name] = df[col].shift(lag)
        lag_cols.append(lag_col_name)
    lag_columns[lag] = lag_cols

# Drop rows with NaN from lagging (first max_lag rows)
df_model = df.dropna().reset_index(drop=True)
print(f'Rows after dropping NaN from lagging: {len(df_model)}')

# ── 5. kNN lag analysis — find optimal lag window ──
print(f'\n── kNN Lag Analysis ──')
print(f'Testing which lag windows best predict NIFTY UP/DOWN...\n')

y = df_model['NIFTY_Direction'].values
results = []

# Test individual lags (1 through max_lag)
for lag in range(1, max_lag + 1):
    feature_cols = lag_columns[lag]
    X = df_model[feature_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test multiple k values
    for k in [3, 5, 7, 9]:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
        results.append({
            'lag_window': f'lag {lag} only',
            'lag_start': lag,
            'lag_end': lag,
            'k': k,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'n_features': len(feature_cols),
        })
        print(f'  Lag {lag:2d} | k={k} | Accuracy: {scores.mean():.4f} ± {scores.std():.4f}')

# Test cumulative lag windows (lag 1-2, 1-3, ..., 1-max_lag)
print(f'\n  Cumulative lag windows:')
for end_lag in range(1, max_lag + 1):
    feature_cols = []
    for lag in range(1, end_lag + 1):
        feature_cols.extend(lag_columns[lag])
    
    X = df_model[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for k in [3, 5, 7, 9]:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
        results.append({
            'lag_window': f'lag 1-{end_lag}',
            'lag_start': 1,
            'lag_end': end_lag,
            'k': k,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'n_features': len(feature_cols),
        })
        print(f'  Lag 1-{end_lag:2d} | k={k} | Accuracy: {scores.mean():.4f} ± {scores.std():.4f} | Features: {len(feature_cols)}')

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mean_accuracy', ascending=False).reset_index(drop=True)

print(f'\n── Top 10 Lag/k Combinations ──')
print(results_df.head(10).to_string(index=False))

# Best combination
best = results_df.iloc[0]
print(f'\n★ Best: {best["lag_window"]} with k={int(best["k"])} → Accuracy: {best["mean_accuracy"]:.4f}')

# ── 6. Save results ──
# kNN analysis results
knn_path = os.path.join(input_fp, 'knn_lag_analysis.xlsx')
results_df.to_excel(knn_path, index=False)
print(f'\nkNN lag analysis saved to: {knn_path}')

# Save model-ready dataset (with best lag window features)
best_end_lag = int(best['lag_end'])
best_start_lag = int(best['lag_start'])

# Keep: Dates, target, and the best lag features
keep_cols = ['Dates', 'NIFTY_Direction']
for lag in range(best_start_lag, best_end_lag + 1):
    keep_cols.extend(lag_columns[lag])

df_final = df_model[keep_cols].copy()
model_path = os.path.join(input_fp, 'IDX_model_data.xlsx')
df_final.to_excel(model_path, index=False)
print(f'Model-ready data saved to: {model_path} — shape: {df_final.shape}')

# Also save imbalance info
imbalance_path = os.path.join(input_fp, 'imbalance_info.txt')
with open(imbalance_path, 'w') as f:
    f.write(f'Target: NIFTY_Direction (1=UP, 0=DOWN)\n')
    f.write(f'UP   (1): {up_count}  ({up_count/len(df)*100:.1f}%)\n')
    f.write(f'DOWN (0): {down_count}  ({down_count/len(df)*100:.1f}%)\n')
    f.write(f'Imbalance ratio (UP/DOWN): {imbalance_ratio:.3f}\n')
    f.write(f'\nBest lag window: {best["lag_window"]}\n')
    f.write(f'Best k: {int(best["k"])}\n')
    f.write(f'Best accuracy: {best["mean_accuracy"]:.4f}\n')
print(f'Imbalance info saved to: {imbalance_path}')
