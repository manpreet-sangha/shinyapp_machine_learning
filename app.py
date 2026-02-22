"""
app.py — Shiny Core App
────────────────────────
Interactive Shiny app to visualise tree-based methods for
predicting NIFTY 50 direction (UP / DOWN) using global index data.

Designed for a non-technical audience to understand:
  • How tree-based methods work (Decision Tree, Random Forest, Gradient Boosting)
  • Key conclusions from the data analysis

Run with:  shiny run app.py
"""

import pandas as pd
import numpy as np
import os
import io
import base64

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler

# ─── Data paths ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'input_data')

# ─── Load data once at startup ───
df_raw = pd.read_excel(os.path.join(INPUT_DIR, 'IDX_data_filtered.xlsx'))
df_raw['Dates'] = pd.to_datetime(df_raw['Dates'], errors='coerce')

knn_results = pd.read_excel(os.path.join(INPUT_DIR, 'knn_lag_analysis.xlsx'))

# Create target
df_raw['NIFTY_Direction'] = (df_raw['NIFTY_CHG_PCT_1D'] > 0).astype(int)

# Identify CHG_PCT columns
CHG_COLS = [c for c in df_raw.columns if 'CHG_PCT' in c]

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
    'INVIXN': 'India VIX',
    'VXEFA': 'EFA Volatility',
    'VXEEM': 'EM Volatility',
    'V2X': 'Euro Stoxx Vol',
    'STI': 'Straits Times (SG)',
    'VHSI': 'HSI Volatility',
}


# ═══════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════
app_ui = ui.page_navbar(
    ui.nav_spacer(),

    # ── TAB 1: Overview ──
    ui.nav_panel(
        "📊 Data Overview",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("About This App"),
                ui.markdown("""
This app helps you understand **tree-based machine learning methods**
applied to predicting whether the **NIFTY 50** index (India's leading
stock market index) will go **UP ↑** or **DOWN ↓** on any given day.

We use data from **15 global stock indices** to see if movements in
other markets can help predict NIFTY's direction.

**Navigate the tabs above** to explore each section.
                """),
                ui.hr(),
                ui.h5("Dataset Summary"),
                ui.output_ui("data_summary_card"),
                width=350,
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Class Balance",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Target Variable: NIFTY Direction"),
                            ui.markdown("""
**What are we predicting?**
Each trading day, the NIFTY 50 either goes **UP** (positive % change)
or **DOWN** (negative % change). A balanced dataset means the model
isn't biased toward one outcome.
                            """),
                            output_widget("imbalance_chart"),
                        ),
                        ui.card(
                            ui.card_header("What is Imbalance?"),
                            ui.markdown("""
**Class imbalance** occurs when one outcome is much more common than
the other. For example, if 90% of days were UP, a model could just
always predict UP and be "90% accurate" — but useless!

Our dataset is **well-balanced**, meaning the model has to actually
*learn* patterns to make good predictions.
                            """),
                            ui.output_ui("imbalance_stats"),
                        ),
                        col_widths=[7, 5],
                    ),
                ),
                ui.nav_panel(
                    "Global Markets",
                    ui.card(
                        ui.card_header("How Do Global Markets Move Together?"),
                        ui.markdown("Select indices to see how their daily percentage changes compare over time."),
                        ui.layout_columns(
                            ui.input_selectize(
                                "selected_indices", "Choose indices to compare:",
                                choices={f'{k}_CHG_PCT_1D': v for k, v in INDEX_NAMES.items()},
                                selected=['NIFTY_CHG_PCT_1D', 'DJ_CHG_PCT_1D', 'SP_CHG_PCT_1D'],
                                multiple=True,
                            ),
                            col_widths=[12],
                        ),
                        output_widget("market_timeseries"),
                    ),
                ),
            ),
        ),
    ),

    # ── TAB 2: kNN Lag Analysis ──
    ui.nav_panel(
        "🔍 Lag Analysis (kNN)",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("What Are Lags?"),
                ui.markdown("""
A **lag** means using *yesterday's* data (or 2 days ago, 3 days ago, etc.)
to predict *today's* outcome.

**Why?** Markets don't react instantly — news from the US market last
night might affect India's market today.

We use **k-Nearest Neighbours (kNN)** — a simple method that predicts
based on similar past days — to find the best lag window.
                """),
                ui.hr(),
                ui.input_slider("knn_k_filter", "Filter by k (neighbours):",
                                min=3, max=9, value=[3, 9], step=2),
                ui.input_radio_buttons("lag_type_filter", "Lag type:",
                                       {"all": "All", "single": "Single lags only",
                                        "cumulative": "Cumulative windows only"}),
                width=350,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("kNN Accuracy by Lag Window"),
                    ui.markdown("Higher bars = better prediction. The best lag tells us how far back to look."),
                    output_widget("knn_lag_chart"),
                ),
                ui.card(
                    ui.card_header("Key Findings"),
                    ui.output_ui("knn_findings"),
                ),
                col_widths=[8, 4],
            ),
        ),
    ),

    # ── TAB 3: Decision Tree ──
    ui.nav_panel(
        "🌳 Decision Tree",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("How Decision Trees Work"),
                ui.markdown("""
A **Decision Tree** is like a flowchart of yes/no questions:

1. *"Was the S&P 500 down more than 0.5% yesterday?"*
2. If YES → *"Was the VIX above 20?"*
3. Continue until reaching a prediction: **UP** or **DOWN**

Each split finds the question that best separates UP days from DOWN days.
                """),
                ui.hr(),
                ui.input_slider("dt_max_depth", "Tree depth (complexity):",
                                min=1, max=8, value=3, step=1),
                ui.input_slider("dt_lag", "Lag window (days back):",
                                min=1, max=7, value=5, step=1),
                ui.input_slider("dt_test_size", "Test set size (%):",
                                min=10, max=40, value=20, step=5),
                ui.input_selectize(
                    "dt_feature_prefixes", "Indices to use as features:",
                    choices={k: v for k, v in INDEX_NAMES.items() if k != 'NIFTY'},
                    selected=['DJ', 'SP', 'DAX', 'HSI', 'INVIXN'],
                    multiple=True,
                ),
                width=380,
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Tree Visualisation",
                    ui.markdown("""
**Reading the tree:** Start at the top. At each box, follow the **left branch
if the condition is true**, or the **right branch if false**. The bottom boxes
(leaves) show the prediction — the colour shows UP (green) or DOWN (red).
                    """),
                    output_widget("dt_tree_viz"),
                ),
                ui.nav_panel(
                    "Feature Space",
                    ui.markdown("""
This shows how the tree **partitions the space** — just like the textbook
diagram! Each coloured region is where the tree predicts UP or DOWN.
Select two features to see the 2D view.
                    """),
                    ui.layout_columns(
                        ui.input_selectize("dt_feat_x", "X-axis feature:", choices=[], selected=None),
                        ui.input_selectize("dt_feat_y", "Y-axis feature:", choices=[], selected=None),
                        col_widths=[6, 6],
                    ),
                    output_widget("dt_feature_space"),
                ),
                ui.nav_panel(
                    "Performance",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Confusion Matrix"),
                            ui.markdown("Shows correct vs incorrect predictions. Diagonal = correct."),
                            output_widget("dt_confusion"),
                        ),
                        ui.card(
                            ui.card_header("Metrics"),
                            ui.output_ui("dt_metrics"),
                        ),
                        col_widths=[7, 5],
                    ),
                ),
            ),
        ),
    ),

    # ── TAB 4: Random Forest ──
    ui.nav_panel(
        "🌲 Random Forest",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("What is a Random Forest?"),
                ui.markdown("""
Instead of relying on **one** tree, a **Random Forest** builds
**many trees** (a "forest") and lets them **vote** on the prediction.

Each tree sees a random subset of the data and features, so they
each learn slightly different patterns. The **majority vote** is
usually more accurate and stable than any single tree.
                """),
                ui.hr(),
                ui.input_slider("rf_n_trees", "Number of trees:", min=10, max=300, value=100, step=10),
                ui.input_slider("rf_max_depth", "Max tree depth:", min=1, max=10, value=4, step=1),
                ui.input_slider("rf_lag", "Lag window:", min=1, max=7, value=5, step=1),
                ui.input_slider("rf_test_size", "Test set size (%):", min=10, max=40, value=20, step=5),
                width=350,
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Feature Importance",
                    ui.markdown("""
**Which markets matter most?** The forest tells us which features
(indices / lags) are most useful for predicting NIFTY's direction.
Taller bars = more important.
                    """),
                    output_widget("rf_importance"),
                ),
                ui.nav_panel(
                    "Performance",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Confusion Matrix"),
                            output_widget("rf_confusion"),
                        ),
                        ui.card(
                            ui.card_header("Metrics & Comparison"),
                            ui.output_ui("rf_metrics"),
                        ),
                        col_widths=[7, 5],
                    ),
                ),
                ui.nav_panel(
                    "Trees vs Accuracy",
                    ui.markdown("How does accuracy change as we add more trees to the forest?"),
                    output_widget("rf_learning_curve"),
                ),
            ),
        ),
    ),

    # ── TAB 5: Gradient Boosting ──
    ui.nav_panel(
        "🚀 Gradient Boosting",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("What is Gradient Boosting?"),
                ui.markdown("""
**Gradient Boosting** also builds many trees, but differently:

1. Start with a simple prediction
2. Build a small tree to fix the **mistakes**
3. Build another tree to fix the **remaining mistakes**
4. Repeat — each tree focuses on what previous trees got wrong

This **sequential learning** often gives the best accuracy.
                """),
                ui.hr(),
                ui.input_slider("gb_n_trees", "Number of stages:", min=10, max=300, value=100, step=10),
                ui.input_slider("gb_max_depth", "Max tree depth:", min=1, max=6, value=3, step=1),
                ui.input_slider("gb_learning_rate", "Learning rate:", min=0.01, max=0.5, value=0.1, step=0.01),
                ui.input_slider("gb_lag", "Lag window:", min=1, max=7, value=5, step=1),
                ui.input_slider("gb_test_size", "Test set size (%):", min=10, max=40, value=20, step=5),
                width=350,
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Feature Importance",
                    output_widget("gb_importance"),
                ),
                ui.nav_panel(
                    "Performance",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Confusion Matrix"),
                            output_widget("gb_confusion"),
                        ),
                        ui.card(
                            ui.card_header("Metrics"),
                            ui.output_ui("gb_metrics"),
                        ),
                        col_widths=[7, 5],
                    ),
                ),
                ui.nav_panel(
                    "Staged Accuracy",
                    ui.markdown("""
Watch how prediction accuracy improves as more trees are added.
Unlike Random Forest (independent trees), each boosting stage
**builds on the previous one**.
                    """),
                    output_widget("gb_staged"),
                ),
            ),
        ),
    ),

    # ── TAB 6: Model Comparison ──
    ui.nav_panel(
        "⚖️ Compare Models",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Side-by-Side Comparison"),
                ui.markdown("""
Compare all three tree-based methods on the same data to see
which works best for predicting NIFTY direction.

Adjust the settings and click **Run Comparison** to update.
                """),
                ui.hr(),
                ui.input_slider("cmp_lag", "Lag window:", min=1, max=7, value=5, step=1),
                ui.input_slider("cmp_test_size", "Test set (%):", min=10, max=40, value=20, step=5),
                ui.input_action_button("cmp_run", "Run Comparison", class_="btn-primary btn-lg w-100"),
                width=320,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Accuracy Comparison"),
                    output_widget("cmp_accuracy"),
                ),
                ui.card(
                    ui.card_header("Key Takeaways"),
                    ui.output_ui("cmp_takeaways"),
                ),
                col_widths=[7, 5],
            ),
        ),
    ),

    title="NIFTY 50 Direction Predictor — Tree-Based ML Methods",
    id="main_nav",
    navbar_options=ui.navbar_options(bg="#1a1a2e", theme="dark"),
)


# ═══════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════

def build_lag_features(df, lag_window, feature_prefixes=None):
    """Build lagged CHG_PCT features and return X, y, feature_names."""
    chg_cols = [c for c in df.columns if 'CHG_PCT' in c]
    if feature_prefixes:
        chg_cols = [c for c in chg_cols
                    if any(c.startswith(p + '_') for p in feature_prefixes)]

    lag_dfs = []
    feature_names = []
    for lag in range(1, lag_window + 1):
        for col in chg_cols:
            lag_name = f'{col}_lag{lag}'
            lag_dfs.append(df[col].shift(lag).rename(lag_name))
            feature_names.append(lag_name)

    temp = pd.concat([df] + lag_dfs, axis=1)
    temp = temp.dropna().reset_index(drop=True)
    X = temp[feature_names].values
    y = temp['NIFTY_Direction'].values
    return X, y, feature_names, temp


def friendly_name(col_name):
    """Convert 'DJ_CHG_PCT_1D_lag3' to 'Dow Jones 1D Chg (lag 3)'."""
    parts = col_name.split('_')
    # Find prefix
    for prefix, name in INDEX_NAMES.items():
        if col_name.startswith(prefix + '_'):
            rest = col_name[len(prefix) + 1:]
            # Extract lag
            lag_part = ''
            if '_lag' in rest:
                idx = rest.index('_lag')
                lag_part = f' (lag {rest[idx + 4:]})'
                rest = rest[:idx]
            rest = rest.replace('CHG_PCT_', '').replace('_', ' ')
            short_name = name.split('(')[0].strip()
            return f'{short_name} {rest}{lag_part}'
    return col_name


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
    cm = confusion_matrix(y_true, y_pred)

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
            ui.tags.tr(ui.tags.td("Precision", style="padding:4px 12px;"), ui.tags.td(f"{prec:.3f}", style="padding:4px 12px; font-weight:bold;")),
            ui.tags.tr(ui.tags.td("Recall", style="padding:4px 12px;"), ui.tags.td(f"{rec:.3f}", style="padding:4px 12px; font-weight:bold;")),
            ui.tags.tr(ui.tags.td("F1 Score", style="padding:4px 12px;"), ui.tags.td(f"{f1:.3f}", style="padding:4px 12px; font-weight:bold;")),
            style="width:100%; border-collapse:collapse;",
        ),
        ui.hr(),
        ui.markdown(f"""
**What do these mean?**
- **Accuracy**: {acc:.1%} of predictions were correct
- **Precision**: When it predicted UP, it was right {prec:.1%} of the time
- **Recall**: It caught {rec:.1%} of actual UP days
        """),
    )


# ═══════════════════════════════════════════════════════════
#  Server
# ═══════════════════════════════════════════════════════════

def server(input: Inputs, output: Outputs, session: Session):

    # ── Data summary card ──
    @output
    @render.ui
    def data_summary_card():
        n = len(df_raw)
        up = df_raw['NIFTY_Direction'].sum()
        down = n - up
        return ui.div(
            ui.tags.table(
                ui.tags.tr(ui.tags.td("Rows:"), ui.tags.td(f"{n}", style="font-weight:bold;")),
                ui.tags.tr(ui.tags.td("Columns:"), ui.tags.td(f"{len(df_raw.columns)}", style="font-weight:bold;")),
                ui.tags.tr(ui.tags.td("Date range:"), ui.tags.td(f"{df_raw['Dates'].min().date()} to {df_raw['Dates'].max().date()}", style="font-weight:bold;")),
                ui.tags.tr(ui.tags.td("Indices:"), ui.tags.td("15 global", style="font-weight:bold;")),
                ui.tags.tr(ui.tags.td("UP days:"), ui.tags.td(f"{up} ({up/n*100:.1f}%)", style="font-weight:bold; color:#166534;")),
                ui.tags.tr(ui.tags.td("DOWN days:"), ui.tags.td(f"{down} ({down/n*100:.1f}%)", style="font-weight:bold; color:#dc2626;")),
                style="width:100%;",
            ),
        )

    # ── Imbalance chart ──
    @render_widget
    def imbalance_chart():
        up = int(df_raw['NIFTY_Direction'].sum())
        down = len(df_raw) - up
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['DOWN ↓'], y=[down], marker_color='#ef4444',
                             text=[f'{down}<br>({down/len(df_raw)*100:.1f}%)'],
                             textposition='inside', textfont=dict(size=16, color='white')))
        fig.add_trace(go.Bar(x=['UP ↑'], y=[up], marker_color='#22c55e',
                             text=[f'{up}<br>({up/len(df_raw)*100:.1f}%)'],
                             textposition='inside', textfont=dict(size=16, color='white')))
        fig.update_layout(
            showlegend=False, height=350,
            yaxis_title='Number of Trading Days',
            margin=dict(l=50, r=20, t=20, b=40),
        )
        return fig

    @output
    @render.ui
    def imbalance_stats():
        up = df_raw['NIFTY_Direction'].sum()
        down = len(df_raw) - up
        ratio = up / down if down > 0 else 0
        return ui.div(
            ui.div(
                ui.h4("Imbalance Ratio"),
                ui.div(f"{ratio:.3f}", style="font-size:2.5em; font-weight:bold; color:#2563eb; text-align:center;"),
                ui.p("(UP count ÷ DOWN count)", style="text-align:center; color:#666;"),
                style="background:#eff6ff; border-radius:8px; padding:16px; margin-bottom:12px;",
            ),
            ui.markdown(f"""
A ratio of **{ratio:.3f}** means the classes are almost perfectly balanced.
This is great for machine learning — the model can't "cheat" by always
guessing one direction.

✅ No resampling or class weighting needed.
            """),
        )

    # ── Global markets time series ──
    @render_widget
    def market_timeseries():
        selected = input.selected_indices()
        if not selected:
            fig = go.Figure()
            fig.update_layout(title="Select at least one index above")
            return fig

        fig = go.Figure()
        for col in selected:
            prefix = col.split('_CHG_PCT')[0]
            name = INDEX_NAMES.get(prefix, prefix)
            fig.add_trace(go.Scatter(
                x=df_raw['Dates'], y=df_raw[col],
                mode='lines', name=name, opacity=0.8,
            ))
        fig.update_layout(
            height=450,
            yaxis_title='Daily % Change',
            xaxis_title='Date',
            legend=dict(orientation='h', y=-0.15),
            margin=dict(l=50, r=20, t=20, b=80),
            hovermode='x unified',
        )
        return fig

    # ── kNN lag chart ──
    @render_widget
    def knn_lag_chart():
        df_k = knn_results.copy()

        # Filter by k range
        k_min, k_max = input.knn_k_filter()
        df_k = df_k[(df_k['k'] >= k_min) & (df_k['k'] <= k_max)]

        # Filter by lag type
        lag_type = input.lag_type_filter()
        if lag_type == 'single':
            df_k = df_k[df_k['lag_window'].str.contains('only')]
        elif lag_type == 'cumulative':
            df_k = df_k[~df_k['lag_window'].str.contains('only')]

        # Group by lag_window, take best k for each
        best_per_window = df_k.loc[df_k.groupby('lag_window')['mean_accuracy'].idxmax()]
        best_per_window = best_per_window.sort_values('mean_accuracy', ascending=True)

        colors = ['#22c55e' if v == best_per_window['mean_accuracy'].max() else '#3b82f6'
                  for v in best_per_window['mean_accuracy']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=best_per_window['lag_window'],
            x=best_per_window['mean_accuracy'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:.1%} (k={int(k)})" for v, k in
                  zip(best_per_window['mean_accuracy'], best_per_window['k'])],
            textposition='outside',
        ))
        fig.add_vline(x=0.5, line_dash='dash', line_color='red',
                      annotation_text='50% (random)', annotation_position='top left')
        fig.update_layout(
            height=max(400, len(best_per_window) * 28),
            xaxis_title='Accuracy (5-fold CV)',
            xaxis_range=[0.4, 0.65],
            margin=dict(l=100, r=80, t=20, b=50),
        )
        return fig

    @output
    @render.ui
    def knn_findings():
        best = knn_results.iloc[0]
        return ui.div(
            ui.div(
                ui.h5("🏆 Best Lag Window"),
                ui.div(f"{best['lag_window']}", style="font-size:1.5em; font-weight:bold; color:#2563eb;"),
                ui.p(f"with k = {int(best['k'])} neighbours"),
                ui.div(f"{best['mean_accuracy']:.1%}", style="font-size:2em; font-weight:bold; color:#166534;"),
                ui.p("accuracy (5-fold CV)"),
                style="background:#f0fdf4; border-radius:8px; padding:16px; text-align:center;",
            ),
            ui.hr(),
            ui.markdown(f"""
**What this means:**

Looking at the **past {int(best['lag_end'])} trading days** of global
market movements gives the best signal for predicting NIFTY's direction.

The accuracy of **{best['mean_accuracy']:.1%}** is above the 50% random
baseline, suggesting there *is* some predictable pattern in how global
markets influence NIFTY — though markets are inherently hard to predict.
            """),
        )

    # ── Decision Tree: update feature selectors ──
    @reactive.Effect
    @reactive.event(input.dt_lag, input.dt_feature_prefixes)
    def _update_dt_features():
        lag = input.dt_lag()
        prefixes = list(input.dt_feature_prefixes()) if input.dt_feature_prefixes() else ['DJ', 'SP']
        chg_cols = [c for c in df_raw.columns if 'CHG_PCT' in c
                    and any(c.startswith(p + '_') for p in prefixes)]
        feature_names = []
        for l in range(1, lag + 1):
            for col in chg_cols:
                feature_names.append(f'{col}_lag{l}')
        choices = {f: friendly_name(f) for f in feature_names[:50]}
        first_two = list(choices.keys())[:2] if len(choices) >= 2 else list(choices.keys())
        ui.update_selectize("dt_feat_x", choices=choices,
                            selected=first_two[0] if first_two else None)
        ui.update_selectize("dt_feat_y", choices=choices,
                            selected=first_two[1] if len(first_two) > 1 else None)

    # ── Decision Tree reactive model ──
    @reactive.Calc
    def dt_model_data():
        lag = input.dt_lag()
        depth = input.dt_max_depth()
        test_pct = input.dt_test_size() / 100
        prefixes = list(input.dt_feature_prefixes()) if input.dt_feature_prefixes() else ['DJ', 'SP']

        X, y, feat_names, df_temp = build_lag_features(df_raw, lag, prefixes)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_pct, random_state=42, shuffle=False,
        )
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return clf, X_train, X_test, y_train, y_test, y_pred, feat_names, df_temp

    # ── Decision Tree visualisation ──
    @render_widget
    def dt_tree_viz():
        clf, X_train, X_test, y_train, y_test, y_pred, feat_names, _ = dt_model_data()
        tree = clf.tree_
        friendly_names = [friendly_name(f) for f in feat_names]

        # Build tree structure for plotly
        def get_tree_data(node, x, y, dx, depth=0):
            nodes_data = []
            edges_data = []

            n_samples = tree.n_node_samples[node]
            values = tree.value[node][0]
            majority = 'UP ↑' if values[1] >= values[0] else 'DOWN ↓'
            color = '#22c55e' if values[1] >= values[0] else '#ef4444'

            if tree.children_left[node] == -1:  # Leaf
                label = f'<b>{majority}</b><br>{int(values[0])}D / {int(values[1])}U<br>n={n_samples}'
            else:
                feat_idx = tree.feature[node]
                threshold = tree.threshold[node]
                fname = friendly_names[feat_idx] if feat_idx < len(friendly_names) else f'Feature {feat_idx}'
                label = f'<b>{fname}</b><br>≤ {threshold:.3f}?<br>n={n_samples}'

            nodes_data.append((x, y, label, color, tree.children_left[node] == -1))

            if tree.children_left[node] != -1:
                left = tree.children_left[node]
                right = tree.children_right[node]
                new_dx = dx / 2

                # Left child
                lx, ly = x - dx, y - 1
                edges_data.append((x, y, lx, ly, 'Yes'))
                ln, le = get_tree_data(left, lx, ly, new_dx, depth + 1)
                nodes_data.extend(ln)
                edges_data.extend(le)

                # Right child
                rx, ry = x + dx, y - 1
                edges_data.append((x, y, rx, ry, 'No'))
                rn, re = get_tree_data(right, rx, ry, new_dx, depth + 1)
                nodes_data.extend(rn)
                edges_data.extend(re)

            return nodes_data, edges_data

        nodes, edges = get_tree_data(0, 0, 0, 2 ** (clf.get_depth() - 1) if clf.get_depth() > 0 else 1)

        fig = go.Figure()

        # Draw edges
        for x1, y1, x2, y2, lbl in edges:
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2], mode='lines',
                line=dict(color='#9ca3af', width=1.5),
                showlegend=False, hoverinfo='skip',
            ))
            fig.add_annotation(x=(x1 + x2) / 2, y=(y1 + y2) / 2,
                               text=lbl, showarrow=False,
                               font=dict(size=10, color='#6b7280'))

        # Draw nodes
        for x, y, label, color, is_leaf in nodes:
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text',
                marker=dict(size=40 if is_leaf else 50,
                            color=color if is_leaf else '#f8fafc',
                            line=dict(color=color, width=2),
                            symbol='square'),
                text=label, textposition='middle center',
                textfont=dict(size=9),
                showlegend=False, hoverinfo='text',
                hovertext=label.replace('<br>', '\n').replace('<b>', '').replace('</b>', ''),
            ))

        fig.update_layout(
            height=max(400, (clf.get_depth() + 1) * 120),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='white',
        )
        return fig

    # ── Decision Tree feature space ──
    @render_widget
    def dt_feature_space():
        clf, X_train, X_test, y_train, y_test, y_pred, feat_names, df_temp = dt_model_data()

        fx = input.dt_feat_x()
        fy = input.dt_feat_y()

        if not fx or not fy or fx not in feat_names or fy not in feat_names:
            fig = go.Figure()
            fig.update_layout(title="Select two features above")
            return fig

        fi_x = feat_names.index(fx)
        fi_y = feat_names.index(fy)

        # Create mesh grid
        x_all = np.concatenate([X_train[:, fi_x], X_test[:, fi_x]])
        y_all = np.concatenate([X_train[:, fi_y], X_test[:, fi_y]])
        x_min, x_max = x_all.min() - 0.5, x_all.max() + 0.5
        y_min, y_max = y_all.min() - 0.5, y_all.max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 150),
            np.linspace(y_min, y_max, 150),
        )
        # Build grid input: set all features to median except the two selected
        medians = np.median(X_train, axis=0)
        grid_input = np.tile(medians, (xx.ravel().shape[0], 1))
        grid_input[:, fi_x] = xx.ravel()
        grid_input[:, fi_y] = yy.ravel()

        Z = clf.predict(grid_input).reshape(xx.shape)

        fig = go.Figure()

        # Decision regions
        fig.add_trace(go.Contour(
            z=Z, x=np.linspace(x_min, x_max, 150),
            y=np.linspace(y_min, y_max, 150),
            colorscale=[[0, 'rgba(239,68,68,0.25)'], [1, 'rgba(34,197,94,0.25)']],
            showscale=False, contours=dict(showlines=True, coloring='fill'),
            hoverinfo='skip',
        ))

        # Test points
        y_test_full = np.concatenate([y_train, y_test])
        X_full = np.concatenate([X_train, X_test])
        colors = ['#22c55e' if v == 1 else '#ef4444' for v in y_test_full]
        symbols = ['UP ↑' if v == 1 else 'DOWN ↓' for v in y_test_full]

        fig.add_trace(go.Scatter(
            x=X_full[:, fi_x], y=X_full[:, fi_y],
            mode='markers',
            marker=dict(color=colors, size=5, opacity=0.6,
                        line=dict(width=0.5, color='white')),
            text=symbols, hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}',
            showlegend=False,
        ))

        fig.update_layout(
            xaxis_title=friendly_name(fx),
            yaxis_title=friendly_name(fy),
            height=450,
            margin=dict(l=60, r=20, t=20, b=60),
        )
        return fig

    # ── Decision Tree confusion & metrics ──
    @render_widget
    def dt_confusion():
        _, _, _, _, y_test, y_pred, _, _ = dt_model_data()
        return make_confusion_fig(y_test, y_pred, 'Decision Tree')

    @output
    @render.ui
    def dt_metrics():
        _, _, _, _, y_test, y_pred, _, _ = dt_model_data()
        return metrics_html(y_test, y_pred, 'Decision Tree')

    # ── Random Forest reactive model ──
    @reactive.Calc
    def rf_model_data():
        lag = input.rf_lag()
        depth = input.rf_max_depth()
        n_trees = input.rf_n_trees()
        test_pct = input.rf_test_size() / 100

        X, y, feat_names, _ = build_lag_features(df_raw, lag)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_pct, random_state=42, shuffle=False,
        )
        clf = RandomForestClassifier(
            n_estimators=n_trees, max_depth=depth, random_state=42, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return clf, X_train, X_test, y_train, y_test, y_pred, feat_names

    @render_widget
    def rf_importance():
        clf, _, _, _, _, _, feat_names = rf_model_data()
        importances = clf.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[friendly_name(feat_names[i]) for i in indices],
            x=importances[indices],
            orientation='h',
            marker_color='#3b82f6',
        ))
        fig.update_layout(
            height=500,
            xaxis_title='Importance (Gini)',
            margin=dict(l=200, r=20, t=20, b=50),
        )
        return fig

    @render_widget
    def rf_confusion():
        _, _, _, _, y_test, y_pred, _ = rf_model_data()
        return make_confusion_fig(y_test, y_pred, 'Random Forest')

    @output
    @render.ui
    def rf_metrics():
        _, _, _, _, y_test, y_pred, _ = rf_model_data()
        return metrics_html(y_test, y_pred, 'Random Forest')

    @render_widget
    def rf_learning_curve():
        lag = input.rf_lag()
        max_trees = input.rf_n_trees()
        depth = input.rf_max_depth()
        test_pct = input.rf_test_size() / 100

        X, y, _, _ = build_lag_features(df_raw, lag)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_pct, random_state=42, shuffle=False,
        )

        tree_counts = list(range(10, max_trees + 1, 10))
        train_accs = []
        test_accs = []
        for n in tree_counts:
            rf = RandomForestClassifier(n_estimators=n, max_depth=depth,
                                        random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            train_accs.append(accuracy_score(y_train, rf.predict(X_train)))
            test_accs.append(accuracy_score(y_test, rf.predict(X_test)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tree_counts, y=train_accs, mode='lines+markers',
                                 name='Train', line=dict(color='#3b82f6')))
        fig.add_trace(go.Scatter(x=tree_counts, y=test_accs, mode='lines+markers',
                                 name='Test', line=dict(color='#ef4444')))
        fig.add_hline(y=0.5, line_dash='dash', line_color='gray',
                      annotation_text='50% baseline')
        fig.update_layout(
            height=400,
            xaxis_title='Number of Trees',
            yaxis_title='Accuracy',
            legend=dict(orientation='h', y=-0.15),
            margin=dict(l=50, r=20, t=20, b=60),
        )
        return fig

    # ── Gradient Boosting reactive model ──
    @reactive.Calc
    def gb_model_data():
        lag = input.gb_lag()
        depth = input.gb_max_depth()
        n_trees = input.gb_n_trees()
        lr = input.gb_learning_rate()
        test_pct = input.gb_test_size() / 100

        X, y, feat_names, _ = build_lag_features(df_raw, lag)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_pct, random_state=42, shuffle=False,
        )
        clf = GradientBoostingClassifier(
            n_estimators=n_trees, max_depth=depth,
            learning_rate=lr, random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return clf, X_train, X_test, y_train, y_test, y_pred, feat_names

    @render_widget
    def gb_importance():
        clf, _, _, _, _, _, feat_names = gb_model_data()
        importances = clf.feature_importances_
        indices = np.argsort(importances)[-20:]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[friendly_name(feat_names[i]) for i in indices],
            x=importances[indices],
            orientation='h',
            marker_color='#f59e0b',
        ))
        fig.update_layout(
            height=500,
            xaxis_title='Importance',
            margin=dict(l=200, r=20, t=20, b=50),
        )
        return fig

    @render_widget
    def gb_confusion():
        _, _, _, _, y_test, y_pred, _ = gb_model_data()
        return make_confusion_fig(y_test, y_pred, 'Gradient Boosting')

    @output
    @render.ui
    def gb_metrics():
        _, _, _, _, y_test, y_pred, _ = gb_model_data()
        return metrics_html(y_test, y_pred, 'Gradient Boosting')

    @render_widget
    def gb_staged():
        clf, X_train, X_test, y_train, y_test, _, _ = gb_model_data()

        # Staged predictions
        train_scores = []
        test_scores = []
        stages = list(range(1, clf.n_estimators + 1))

        for i, y_pred_train in enumerate(clf.staged_predict(X_train)):
            train_scores.append(accuracy_score(y_train, y_pred_train))
        for i, y_pred_test in enumerate(clf.staged_predict(X_test)):
            test_scores.append(accuracy_score(y_test, y_pred_test))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stages, y=train_scores, mode='lines',
                                 name='Train', line=dict(color='#f59e0b')))
        fig.add_trace(go.Scatter(x=stages, y=test_scores, mode='lines',
                                 name='Test', line=dict(color='#ef4444')))
        fig.add_hline(y=0.5, line_dash='dash', line_color='gray',
                      annotation_text='50% baseline')
        fig.update_layout(
            height=400,
            xaxis_title='Number of Boosting Stages',
            yaxis_title='Accuracy',
            legend=dict(orientation='h', y=-0.15),
            margin=dict(l=50, r=20, t=20, b=60),
        )
        return fig

    # ── Model Comparison ──
    @reactive.Calc
    @reactive.event(input.cmp_run)
    def comparison_data():
        lag = input.cmp_lag()
        test_pct = input.cmp_test_size() / 100

        X, y, feat_names, _ = build_lag_features(df_raw, lag)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_pct, random_state=42, shuffle=False,
        )

        results = {}
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4,
                                                     random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                             learning_rate=0.1, random_state=42),
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'cv_mean': cv.mean(),
                'cv_std': cv.std(),
            }

        return results

    @render_widget
    def cmp_accuracy():
        results = comparison_data()
        if not results:
            fig = go.Figure()
            fig.update_layout(title="Click 'Run Comparison' to start")
            return fig

        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        colors = {'accuracy': '#3b82f6', 'precision': '#22c55e',
                  'recall': '#f59e0b', 'f1': '#8b5cf6'}

        fig = go.Figure()
        for metric in metrics:
            vals = [results[m][metric] for m in models]
            fig.add_trace(go.Bar(
                name=metric.capitalize(), x=models, y=vals,
                marker_color=colors[metric],
                text=[f'{v:.1%}' for v in vals],
                textposition='outside',
            ))
        fig.add_hline(y=0.5, line_dash='dash', line_color='red',
                      annotation_text='50% baseline')
        fig.update_layout(
            barmode='group', height=450,
            yaxis_title='Score', yaxis_range=[0, 1],
            legend=dict(orientation='h', y=-0.15),
            margin=dict(l=50, r=20, t=20, b=60),
        )
        return fig

    @output
    @render.ui
    def cmp_takeaways():
        results = comparison_data()
        if not results:
            return ui.p("Click 'Run Comparison' to generate results.")

        best_model = max(results, key=lambda m: results[m]['accuracy'])
        best_acc = results[best_model]['accuracy']
        best_cv = results[best_model]['cv_mean']

        return ui.div(
            ui.div(
                ui.h5("🏆 Best Model"),
                ui.div(best_model, style="font-size:1.4em; font-weight:bold; color:#2563eb;"),
                ui.div(f"Test Accuracy: {best_acc:.1%}", style="font-size:1.2em; color:#166534;"),
                ui.div(f"Cross-Val: {best_cv:.1%} ± {results[best_model]['cv_std']:.1%}",
                       style="color:#666;"),
                style="background:#eff6ff; border-radius:8px; padding:16px; text-align:center; margin-bottom:12px;",
            ),
            ui.hr(),
            ui.h5("All Results"),
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Model", style="padding:6px;"),
                        ui.tags.th("Accuracy", style="padding:6px;"),
                        ui.tags.th("CV Mean", style="padding:6px;"),
                    ),
                ),
                ui.tags.tbody(
                    *[ui.tags.tr(
                        ui.tags.td(m, style="padding:6px;"),
                        ui.tags.td(f"{results[m]['accuracy']:.1%}",
                                   style=f"padding:6px; font-weight:bold; color:{'#166534' if m == best_model else '#333'};"),
                        ui.tags.td(f"{results[m]['cv_mean']:.1%}", style="padding:6px;"),
                    ) for m in results],
                ),
                style="width:100%; border-collapse:collapse;",
            ),
            ui.hr(),
            ui.markdown("""
**Key Takeaways:**

1. **Decision Trees** are simple and interpretable but can overfit
2. **Random Forests** reduce overfitting by averaging many trees
3. **Gradient Boosting** learns from mistakes sequentially and often achieves the best accuracy
4. All models beat the 50% random baseline, confirming that global market patterns do contain predictive signals for NIFTY
            """),
        )


# ═══════════════════════════════════════════════════════════
#  Create app
# ═══════════════════════════════════════════════════════════
app = App(app_ui, server)
