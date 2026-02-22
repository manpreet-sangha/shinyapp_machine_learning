# SMM748 Machine Learning in Quantitative Finance — Group Coursework 1

## Shiny App: Visualising Tree-Based Methods for Financial Index Analysis

### 🌐 Live App

**https://manpreet-sangha.shinyapps.io/nifty-ml-predictor/**

### Overview

This project is part of the **SMM748 Machine Learning in Quantitative Finance** module (Term 2) at **City St George's, University of London**. The coursework involves designing and developing an interactive **Python Shiny app** that visualises **tree-based methods** (Decision Trees, Random Forests, Gradient Boosting) applied to global financial index data.

The app predicts whether the **NIFTY 50** index (India) will go **UP ↑** or **DOWN ↓** on any given day, using lagged percentage-change data from 15 global indices. It is designed for a **non-technical audience** — clear, interactive, and accessible.

### Dataset

The dataset consists of daily observations for **15 global financial indices** sourced from Bloomberg Terminal, covering the period **2024-02-14 to 2026-02-20** (after preprocessing).

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
| India VIX | INVIXN | NSE India |
| CBOE EFA Volatility | VXEFA | CBOE |
| CBOE EM Volatility | VXEEM | CBOE |
| EURO STOXX 50 Volatility | V2X | Eurex |
| Straits Times Index | STI | SGX (Singapore) |
| HSI Volatility | VHSI | HKEX |

**Four fields per index:**
- `PX_OPEN` — Opening price
- `PX_CLOSE_1D` — Previous day's closing price
- `CHG_PCT_1D` — 1-day percentage change
- `CHG_PCT_5D` — 5-day percentage change

**Final dataset dimensions:** 499 rows × 61 columns (< 500 observations, < 50 features per index — satisfying coursework requirements).

### Project Structure

```
├── app.py                         # Shiny Core app (main application)
├── 1_data_preprocessing.py        # Step 1: Parse raw Bloomberg data, rename columns
├── 2_data_preprocessing.py        # Step 2: Remove NIFTY holidays, apply date filter
├── 3_feature_engineering.py       # Step 3: Lag features, kNN lag analysis, imbalance check
├── create_trading_holidays.py     # Generate Trading_Holidays.xlsx (10 exchange calendars)
├── requirements.txt               # Python dependencies
├── input_data/
│   ├── IDX_data_v1.xlsx           # Raw Bloomberg data (15 indices, 4 fields each)
│   ├── IDX_data_preprocessed.xlsx # Output of Step 1 (cleaned column names)
│   ├── IDX_data_filtered.xlsx     # Output of Step 2 (final 499-row dataset)
│   ├── IDX_model_data.xlsx        # Output of Step 3 (model-ready with best lag features)
│   ├── knn_lag_analysis.xlsx      # kNN results across all lag/k combinations
│   ├── imbalance_info.txt         # Target class balance summary
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
- Generates a detailed log (`holiday_filter_log.txt`) with all removed dates and holiday names
- Saves output: `IDX_data_filtered.xlsx`

#### Trading Holidays — `create_trading_holidays.py`
- Generates `Trading_Holidays.xlsx` with one sheet per exchange covering 2023–2026
- 10 exchange holiday calendars: NIFTY, DJ, SP, DAX, UKX, HSI, SHCOMP, TWSE, NKY, STI

#### Step 3 — `3_feature_engineering.py`
- Creates binary target variable: `NIFTY_Direction` (1 = UP, 0 = DOWN) from `NIFTY_CHG_PCT_1D`
- Computes **imbalance ratio** — dataset is well-balanced (51.3% UP / 48.7% DOWN, ratio 1.053)
- Builds **lagged features** (lag 1–10) for all 30 `CHG_PCT` columns
- Runs **kNN analysis** across all lag windows and k values (3, 5, 7, 9) with 5-fold cross-validation
- Best result: **lag 1–7 with k=9 → 56.0% accuracy** (above 50% random baseline)
- Saves: `knn_lag_analysis.xlsx`, `IDX_model_data.xlsx`, `imbalance_info.txt`

### Shiny App — `app.py`

Interactive Python Shiny Core application with 6 tabs:

| Tab | Description |
|-----|-------------|
| 📊 **Data Overview** | Dataset summary, class balance visualisation, global market time series |
| 🔍 **Lag Analysis (kNN)** | Interactive exploration of which lag windows best predict NIFTY direction |
| 🌳 **Decision Tree** | Adjustable tree depth, lag, features; tree visualisation, feature space partitions, performance metrics |
| 🌲 **Random Forest** | Configurable number of trees; feature importance, confusion matrix, learning curve |
| 🚀 **Gradient Boosting** | Tunable stages, depth, learning rate; staged accuracy, feature importance |
| ⚖️ **Compare Models** | Side-by-side comparison of all three methods with cross-validation |

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

# Step 2: Filter holidays and trim date range
python 2_data_preprocessing.py

# Step 3: Feature engineering & kNN lag analysis
python 3_feature_engineering.py

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
