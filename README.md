# SMM748 Machine Learning in Quantitative Finance — Group Coursework 1

## Shiny App: Visualising Tree-Based Methods for Financial Index Analysis

### 🌐 Live App

**https://manpreet-sangha.shinyapps.io/nifty-ml-predictor/**

### Overview

This project is part of the **SMM748 Machine Learning in Quantitative Finance** module (Term 2) at **City St George's, University of London**. The coursework involves designing and developing an interactive **Python Shiny app** that visualises **tree-based methods** (Decision Trees, Random Forests, Gradient Boosting) applied to global financial index data.

The app predicts whether the **NIFTY 50** index (India) will go **UP ↑** or **DOWN ↓** on any given day, using **previous-day direction indicators** (binary: UP/DOWN) from **10 global stock exchanges**. This lag-1 strategy avoids look-ahead bias — we only use information available *before* the trading day begins. The app is designed for a **non-technical audience** — clear, interactive, and accessible.

### Dataset

The dataset consists of daily observations for **10 global financial indices** sourced from Bloomberg Terminal, covering the period **2024-02-14 to 2026-02-20** (after preprocessing).

| Index | Ticker | Exchange |
|-------|--------|----------|
| NIFTY 50 | NIFTY | NSE India |
| Dow Jones Industrial Average | INDU (DJ) | NYSE |
| S&P 500 | SPX (SP) | NYSE / CBOE |
| DAX | DAX | XETRA (Germany) |
| FTSE 100 | UKX | LSE (UK) |
| Hang Seng Index | HSI | HKEX (Hong Kong) |
| Shanghai Composite | SHCOMP | SSE (China) |
| Taiwan Weighted Index | TWSE | TWSE (Taiwan) |
| Nikkei 225 | NKY | TSE (Japan) |
| Straits Times Index | STI | SGX (Singapore) |

**Binary encoding:**
- Each index's daily percentage change (`CHG_PCT_1D`) is encoded as:
  - `1` (UP) if the change is positive (> 0)
  - `0` (DOWN) if the change is zero or negative
- VIX / volatility indices are excluded from the final dataset

**Lag-1 strategy:**
- Features = previous day's direction (UP/DOWN) for all 10 exchanges
- Target = today's NIFTY direction
- This avoids look-ahead bias — only yesterday's data is used

**Final dataset dimensions:** 498 usable rows × 10 lag-1 features (after shifting and dropping NaN)

### Project Structure

```
├── app.py                         # Shiny Core app (main application)
├── 1_data_preprocessing.py        # Step 1: Parse raw Bloomberg data, rename columns
├── 2_data_preprocessing.py        # Step 2: Remove NIFTY holidays, apply date filter, keep CHG_PCT_1D columns
├── 3_data_preprocessing.py        # Step 3: Binary encode directions, drop VIX columns
├── create_trading_holidays.py     # Generate Trading_Holidays.xlsx (10 exchange calendars)
├── requirements.txt               # Python dependencies
├── input_data/
│   ├── IDX_data_v1.xlsx           # Raw Bloomberg data (15 indices, 4 fields each)
│   ├── IDX_data_preprocessed.xlsx # Output of Step 1 (cleaned column names)
│   ├── IDX_data_filtered.xlsx     # Output of Step 2 (499 rows, CHG_PCT_1D columns only)
│   ├── IDX_data_encoded.xlsx      # Output of Step 3 (binary direction indicators, 499 rows × 11 cols)
│   ├── Trading_Holidays.xlsx      # Exchange holiday calendars (10 sheets)
│   └── holiday_filter_log.txt     # Detailed log of filtering actions
├── .gitignore
└── README.md
```

### Data Preprocessing Pipeline

#### Step 1 — `1_data_preprocessing.py`
- Reads the raw Bloomberg export (`IDX_data_v1.xlsx`)
- Skips 6 metadata/header rows
- Dynamically detects index names and field labels
- Renames columns with friendly prefixes (e.g. `NIFTY_PX_OPEN`, `DJ_CHG_PCT_1D`, `SP_PX_CLOSE_1D`)
- Converts dates to date-only format
- Saves output: `IDX_data_preprocessed.xlsx`

#### Step 2 — `2_data_preprocessing.py`
- Loads the preprocessed data and `Trading_Holidays.xlsx`
- Removes rows where the Indian market (NIFTY) was closed due to trading holidays (46 rows removed)
- Applies a start-date filter (≥ 2024-02-14) to keep the dataset under 500 rows
- Keeps only `CHG_PCT_1D` columns (drops PX_OPEN, PX_CLOSE_1D, CHG_PCT_5D)
- Generates a detailed log (`holiday_filter_log.txt`) with all removed dates and holiday names
- Saves output: `IDX_data_filtered.xlsx` (499 rows × 16 columns)

#### Trading Holidays — `create_trading_holidays.py`
- Generates `Trading_Holidays.xlsx` with one sheet per exchange covering 2023–2026
- 10 exchange holiday calendars: NIFTY, DJ, SP, DAX, UKX, HSI, SHCOMP, TWSE, NKY, STI

#### Step 3 — `3_data_preprocessing.py`
- Encodes every `CHG_PCT_1D` column as binary: positive change → **1 (UP)**, else → **0 (DOWN)**
- Renames columns from `*_CHG_PCT_1D` → `*_Direction`
- Drops VIX / volatility index columns (INVIXN, VXEFA, VXEEM, V2X, VHSI)
- Prints a summary of UP/DOWN counts per index
- Dataset is well-balanced (e.g. NIFTY: ~51% UP / ~49% DOWN)
- Saves output: `IDX_data_encoded.xlsx` (499 rows × 11 columns: Dates + 10 Direction indicators)

### Shiny App — `app.py`

Interactive Python Shiny Core application with 6 tabs:

| Tab | Description |
|-----|-------------|
| 📊 **Data Overview** | Dataset summary, class balance, conditional bar charts (does yesterday's direction predict NIFTY?), correlation heatmap, global market time series |
| 🔍 **Best Predictor (kNN)** | Tests which exchange's previous-day direction best predicts NIFTY using k-Nearest Neighbours with cross-validation |
| 🌳 **Decision Tree** | Adjustable tree depth and test size; tree visualisation, 2×2 scenario grid (binary feature space), performance metrics, error vs tree size |
| 🌲 **Random Forest** | Configurable number of trees and depth; feature importance, confusion matrix, learning curve |
| 🚀 **Gradient Boosting** | Tunable stages, depth, learning rate; staged accuracy, feature importance |
| ⚖️ **Compare Models** | Side-by-side comparison of all methods with cross-validation |

All explanations are written in **plain language** for a non-technical audience.

### Requirements

- Python 3.11+
- shiny, shinywidgets
- pandas, numpy, openpyxl
- scikit-learn
- plotly, htmltools

Install dependencies:
```bash
pip install -r requirements.txt
```

### How to Run

```bash
# Step 1: Preprocess raw Bloomberg data
python 1_data_preprocessing.py

# Generate trading holidays file (if not already present)
python create_trading_holidays.py

# Step 2: Filter holidays, trim date range, keep CHG_PCT_1D columns
python 2_data_preprocessing.py

# Step 3: Binary encode directions, drop VIX columns
python 3_data_preprocessing.py

# Run the Shiny app locally
shiny run app.py
```

The app will be available at **http://127.0.0.1:8000**

### Deployment

The app is deployed to **ShinyApps.io** and accessible at:

**https://manpreet-sangha.shinyapps.io/nifty-ml-predictor/**

### Assessment Criteria

This coursework is assessed on:
- **Quality of the Shiny app** — clarity of layout, user-friendliness, and how clearly tree-based methods are explained
- **Quality of the report** — presentation, writing, and quality of plots

### Module

**SMM748 — Machine Learning in Quantitative Finance**  
City St George's, University of London  
Term 2, 2025–2026
