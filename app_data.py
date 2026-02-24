"""
app_data.py — Shared Data, Constants & Helper Functions
───────────────────────────────────────────────────────
Loaded once at startup and imported by every component module.
"""

import pandas as pd
import numpy as np
import os
import io
import base64

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from shiny import ui
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
)

# ─── Data paths ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'input_data')

# ─── Load encoded binary data (output of 3_data_preprocessing.py) ───
df_encoded = pd.read_excel(os.path.join(INPUT_DIR, 'IDX_data_encoded.xlsx'))
df_encoded['Dates'] = pd.to_datetime(df_encoded['Dates'], errors='coerce')

# Target = NIFTY_Direction (today)
# Features = previous day's Direction indicators for other exchanges (lag-1)
DIRECTION_COLS = [c for c in df_encoded.columns if c.endswith('_Direction')]
FEATURE_COLS = [c for c in DIRECTION_COLS if c != 'NIFTY_Direction']

# Build lag-1 features: shift each feature column by 1 to use yesterday's values
df_model = df_encoded.copy()
for col in FEATURE_COLS:
    df_model[f'{col}_lag1'] = df_model[col].shift(1)

# Also create NIFTY_Direction_lag1 (yesterday's own movement) as a potential feature
df_model['NIFTY_Direction_lag1'] = df_model['NIFTY_Direction'].shift(1)

# Drop the first row (NaN from shifting)
df_model = df_model.dropna().reset_index(drop=True)

# Feature column names (lag-1 of all exchanges including NIFTY's own lag)
LAG1_FEATURE_COLS = [f'{c}_lag1' for c in FEATURE_COLS] + ['NIFTY_Direction_lag1']

# For backward compatibility, also keep raw data for the overview plots
df_raw_pct = pd.read_excel(os.path.join(INPUT_DIR, 'IDX_data_filtered.xlsx'))
df_raw_pct['Dates'] = pd.to_datetime(df_raw_pct['Dates'], errors='coerce')
df_raw_pct['NIFTY_Direction'] = (df_raw_pct['NIFTY_CHG_PCT_1D'] > 0).astype(int)
CHG_COLS = [c for c in df_raw_pct.columns if 'CHG_PCT' in c]

# All index prefixes for display
INDEX_NAMES = {
    'NIFTY': 'NIFTY 50 (India)',
    'DJ': 'Dow Jones (US)',
    'SP': 'S&P 500 (US)',
    'DAX': 'DAX (Germany)',
    'UKX': 'FTSE 100 (UK)',
    'HSI': 'Hang Seng (HK)',
    'SHCOMP': 'Shanghai (China)',
    'TWSE': 'TWSE (Taiwan)',
    'NKY': 'Nikkei 225 (Japan)',
    'STI': 'Straits Times (SG)',
}


def friendly_name(col_name):
    """Convert 'DJ_Direction_lag1' to 'Dow Jones (prev day)' etc."""
    for prefix, name in INDEX_NAMES.items():
        if col_name.startswith(prefix + '_'):
            short_name = name.split('(')[0].strip()
            if '_lag1' in col_name:
                return f'{short_name} (prev day)'
            return f'{short_name} Direction'
    return col_name


# ═══════════════════════════════════════════════════════════
#  Helper functions used by multiple tabs
# ═══════════════════════════════════════════════════════════

def get_Xy(test_pct=0.2):
    """Return X_train, X_test, y_train, y_test using the pre-built lag-1 features."""
    X = df_model[LAG1_FEATURE_COLS].values
    y = df_model['NIFTY_Direction'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_pct, random_state=42, shuffle=False,
    )
    return X_train, X_test, y_train, y_test


def make_confusion_fig(y_true, y_pred, title=''):
    """Create a plotly confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ['DOWN (0)', 'UP (1)']
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, '#fee2e2'], [1, '#166534']],
        text=cm, texttemplate='%{text}',
        textfont={'size': 20},
        showscale=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=350,
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


def metrics_html(y_true, y_pred, model_name=''):
    """Return an HTML string of classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR = Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR

    return ui.div(
        ui.h5(f'{model_name} Results', style='margin-bottom:12px;'),
        ui.div(
            ui.div(
                ui.span("Accuracy", style="color:#666; font-size:0.85em;"),
                ui.div(f"{acc:.1%}", style="font-size:1.8em; font-weight:bold; color:#166534;"),
                style="text-align:center; padding:10px;",
            ),
            style="background:#f0fdf4; border-radius:8px; margin-bottom:12px; padding:8px;",
        ),
        ui.tags.table(
            ui.tags.tr(
                ui.tags.td("Sensitivity (TPR)", style="padding:4px 12px;"),
                ui.tags.td(f"{sensitivity:.3f}", style="padding:4px 12px; font-weight:bold;"),
            ),
            ui.tags.tr(
                ui.tags.td("Specificity (TNR)", style="padding:4px 12px;"),
                ui.tags.td(f"{specificity:.3f}", style="padding:4px 12px; font-weight:bold;"),
            ),
            ui.tags.tr(
                ui.tags.td("Precision (PPV)", style="padding:4px 12px;"),
                ui.tags.td(f"{prec:.3f}", style="padding:4px 12px; font-weight:bold;"),
            ),
            ui.tags.tr(
                ui.tags.td("F1 Score", style="padding:4px 12px;"),
                ui.tags.td(f"{f1:.3f}", style="padding:4px 12px; font-weight:bold;"),
            ),
            style="width:100%; border-collapse:collapse;",
        ),
        ui.hr(),
        ui.div(
            ui.tags.table(
                ui.tags.caption("Confusion Matrix", style="font-weight:bold; margin-bottom:4px; text-align:left;"),
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("", style="padding:4px 8px;"),
                        ui.tags.th("Pred DOWN", style="padding:4px 8px; color:#dc2626; font-size:0.85em;"),
                        ui.tags.th("Pred UP", style="padding:4px 8px; color:#16a34a; font-size:0.85em;"),
                    ),
                ),
                ui.tags.tbody(
                    ui.tags.tr(
                        ui.tags.td("Actual DOWN", style="padding:4px 8px; font-weight:bold; color:#dc2626; font-size:0.85em;"),
                        ui.tags.td(f"{tn}", style="padding:4px 8px; text-align:center; background:#f0fdf4; font-weight:bold;"),
                        ui.tags.td(f"{fp}", style="padding:4px 8px; text-align:center; background:#fef2f2;"),
                    ),
                    ui.tags.tr(
                        ui.tags.td("Actual UP", style="padding:4px 8px; font-weight:bold; color:#16a34a; font-size:0.85em;"),
                        ui.tags.td(f"{fn}", style="padding:4px 8px; text-align:center; background:#fef2f2;"),
                        ui.tags.td(f"{tp}", style="padding:4px 8px; text-align:center; background:#f0fdf4; font-weight:bold;"),
                    ),
                ),
                style="width:100%; border-collapse:collapse; border:1px solid #e5e7eb;",
            ),
            style="margin-bottom:12px;",
        ),
        ui.hr(),
        ui.markdown(f"""
**What do these numbers mean?**

- **Accuracy** ({acc:.1%}): Out of all test days, how many
  did the model predict correctly?

- **Sensitivity** ({sensitivity:.1%}): When NIFTY actually went
  **UP**, how often did the model catch it?
  *(Measures: can it spot UP days?)*

- **Specificity** ({specificity:.1%}): When NIFTY actually went
  **DOWN**, how often did the model catch it?
  *(Measures: can it spot DOWN days?)*

- **Precision** ({prec:.1%}): When the model said **UP**, how
  often was it actually right?
  *(Measures: can you trust its UP calls?)*

- **F1 Score** ({f1:.3f}): A balanced average of Precision and
  Sensitivity. Higher is better (max = 1.0).

*For trading, **Precision** matters most — you want to trust
the model's signal before placing a trade.*
        """),
    )


def make_roc_fig(y_true, y_prob, model_name=''):
    """Build a Plotly ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'{model_name} (AUC = {roc_auc:.3f})',
        line=dict(color='#2563eb', width=2.5),
        fill='tozeroy', fillcolor='rgba(37,99,235,0.08)',
    ))
    # Random baseline (diagonal)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random guess (AUC = 0.5)',
        line=dict(color='#ef4444', width=1.5, dash='dash'),
    ))

    fig.update_layout(
        title=dict(text=f'{model_name} — ROC Curve', font=dict(size=15)),
        xaxis_title='False Positive Rate (1 - Specificity)',
        yaxis_title='True Positive Rate (Sensitivity)',
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
        height=480,
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(x=0.4, y=0.05),
        plot_bgcolor='white',
    )
    return fig
