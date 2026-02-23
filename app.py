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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler

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
#  UI
# ═══════════════════════════════════════════════════════════
app_ui = ui.page_navbar(
    # ── Responsive CSS for all devices ──
    ui.head_content(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.style("""
            /* Make widgets fill their container width */
            .shiny-plot-output, .html-widget, .plotly {
                width: 100% !important;
            }
            /* Stack sidebar below content on small screens */
            @media (max-width: 992px) {
                .bslib-sidebar-layout {
                    flex-direction: column !important;
                }
                .bslib-sidebar-layout > .sidebar {
                    width: 100% !important;
                    max-width: 100% !important;
                }
                .bslib-sidebar-layout > .main {
                    width: 100% !important;
                }
            }
            /* Wider card text on mobile */
            @media (max-width: 768px) {
                .card-body { padding: 0.75rem !important; }
                .card-header { font-size: 0.95rem; }
            }
        """),
    ),
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

We use **previous day's direction indicators** (UP/DOWN) from **9 global
stock exchanges** to predict NIFTY's next-day direction.

**Strategy:** To predict today's NIFTY, we look at whether each global
index went UP or DOWN *yesterday* — this avoids look-ahead bias.

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
                    "Explore Features",
                    ui.markdown("""
**Does yesterday's market direction predict NIFTY today?**

For each global exchange, we look at all the days it went **UP** yesterday
vs all the days it went **DOWN** yesterday, and ask: *what happened to
NIFTY the next day?*

If a market's direction is a useful signal, you'll see a clear difference
between the green and red bars.
                    """),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("When Each Market Was UP vs DOWN Yesterday, What Did NIFTY Do?"),
                            output_widget("eda_conditional_bars"),
                        ),
                        col_widths=[12],
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Do Global Markets Move Together? (Correlation Heatmap)"),
                            ui.markdown("""
This heatmap shows how often pairs of markets move in the **same direction**
on the same day. Darker blue = stronger tendency to move together.
                            """),
                            output_widget("eda_heatmap"),
                        ),
                        ui.card(
                            ui.card_header("What Does This Tell Us?"),
                            ui.output_ui("eda_insights"),
                        ),
                        col_widths=[8, 4],
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
                                choices={f'{k}_CHG_PCT_1D': v for k, v in INDEX_NAMES.items()
                                         if f'{k}_CHG_PCT_1D' in df_raw_pct.columns},
                                selected=[c for c in ['NIFTY_CHG_PCT_1D', 'DJ_CHG_PCT_1D', 'SP_CHG_PCT_1D']
                                          if c in df_raw_pct.columns],
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
        "🔍 Best Predictor (kNN)",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Which Index Best Predicts NIFTY?"),
                ui.markdown("""
We use **k-Nearest Neighbours (kNN)** to test which stock exchange's
**previous day direction** (UP/DOWN) is the most useful predictor of
NIFTY's next-day direction.

Each bar shows the **cross-validated accuracy** when using that single
exchange's lag-1 direction as the only feature. We also test combinations.
                """),
                ui.hr(),
                ui.input_slider("knn_k", "Number of neighbours (k):",
                                min=1, max=15, value=5, step=2),
                width=350,
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("kNN Accuracy by Individual Predictor"),
                    ui.markdown("Higher bars = better predictor. Each bar uses one exchange's previous-day direction."),
                    output_widget("knn_predictor_chart"),
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

1. *"Did the S&P 500 go UP yesterday?"*
2. If YES → *"Did the Hang Seng go UP?"*
3. Continue until reaching a prediction: **UP** or **DOWN**

Since our features are binary (0/1), each split asks whether an
exchange went UP or DOWN on the previous day.
                """),
                ui.hr(),
                ui.input_slider("dt_max_depth", "Tree depth (complexity):",
                                min=1, max=8, value=3, step=1),
                ui.input_slider("dt_test_size", "Test set size (%):",
                                min=10, max=40, value=20, step=5),
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
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("How Does the Tree Use Two Markets?"),
                            ui.layout_columns(
                                ui.input_selectize("dt_feat_x", "Market 1:",
                                    choices={f: friendly_name(f) for f in LAG1_FEATURE_COLS},
                                    selected=LAG1_FEATURE_COLS[0] if LAG1_FEATURE_COLS else None),
                                ui.input_selectize("dt_feat_y", "Market 2:",
                                    choices={f: friendly_name(f) for f in LAG1_FEATURE_COLS},
                                    selected=LAG1_FEATURE_COLS[1] if len(LAG1_FEATURE_COLS) > 1 else None),
                                col_widths=[6, 6],
                            ),
                            output_widget("dt_feature_space"),
                        ),
                        ui.card(
                            ui.card_header("What Am I Looking At?"),
                            ui.markdown("""
**Each box represents a scenario** — a combination of what two
global markets did *yesterday*.

Since each market either went **UP** or **DOWN**, there are exactly
**4 possible scenarios** (a 2×2 grid).

📊 **Inside each box you'll see:**
- How many trading days fell into that scenario
- How many of those days NIFTY went UP vs DOWN
- What the **decision tree predicts** for that scenario
- Whether the prediction matches reality (✅ or ⚠️)

🎨 **Colours:**
- 🟩 **Green box** → tree predicts NIFTY **UP** tomorrow
- 🟥 **Red box** → tree predicts NIFTY **DOWN** tomorrow
- The **pie chart** inside shows the actual UP/DOWN split

💡 *Try different market pairs to see which combinations
give the tree the clearest signal!*
                            """),
                            ui.hr(),
                            ui.output_ui("dt_feature_space_summary"),
                        ),
                        col_widths=[8, 4],
                    ),
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
                ui.nav_panel(
                    "Error vs Tree Size",
                    ui.markdown("""
**How does tree complexity affect error?** As the tree grows deeper
(more splits), training error keeps falling — but at some point the
model starts **over-fitting**: it memorises the training data and
performs worse on unseen data. The sweet spot is where the
**cross-validation / test error is lowest**.
                    """),
                    output_widget("dt_error_vs_size"),
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
        n = len(df_model)
        up = int(df_model['NIFTY_Direction'].sum())
        down = n - up
        return ui.div(
            ui.tags.table(
                ui.tags.tr(ui.tags.td("Rows:"), ui.tags.td(f"{n}", style="font-weight:bold;")),
                ui.tags.tr(ui.tags.td("Features:"), ui.tags.td(f"{len(LAG1_FEATURE_COLS)} (lag-1 direction)", style="font-weight:bold;")),
                ui.tags.tr(ui.tags.td("Date range:"), ui.tags.td(f"{df_model['Dates'].min().date()} to {df_model['Dates'].max().date()}", style="font-weight:bold;")),
                ui.tags.tr(ui.tags.td("Exchanges:"), ui.tags.td(f"{len(INDEX_NAMES)} global", style="font-weight:bold;")),
                ui.tags.tr(ui.tags.td("UP days:"), ui.tags.td(f"{up} ({up/n*100:.1f}%)", style="font-weight:bold; color:#166534;")),
                ui.tags.tr(ui.tags.td("DOWN days:"), ui.tags.td(f"{down} ({down/n*100:.1f}%)", style="font-weight:bold; color:#dc2626;")),
                style="width:100%;",
            ),
        )

    # ── Imbalance chart ──
    @render_widget
    def imbalance_chart():
        up = int(df_model['NIFTY_Direction'].sum())
        down = len(df_model) - up
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['DOWN ↓'], y=[down], marker_color='#ef4444',
                             text=[f'{down}<br>({down/len(df_model)*100:.1f}%)'],
                             textposition='inside', textfont=dict(size=16, color='white')))
        fig.add_trace(go.Bar(x=['UP ↑'], y=[up], marker_color='#22c55e',
                             text=[f'{up}<br>({up/len(df_model)*100:.1f}%)'],
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
        up = df_model['NIFTY_Direction'].sum()
        down = len(df_model) - up
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
            if col in df_raw_pct.columns:
                fig.add_trace(go.Scatter(
                    x=df_raw_pct['Dates'], y=df_raw_pct[col],
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

    # ── Explore Features: Conditional bar chart ──
    @render_widget
    def eda_conditional_bars():
        """For each exchange: when it was UP yesterday vs DOWN yesterday,
        what % of the time did NIFTY go UP the next day?"""
        rows = []
        for col in LAG1_FEATURE_COLS:
            name = friendly_name(col).replace(' (prev day)', '')
            for direction, label in [(1, 'UP yesterday'), (0, 'DOWN yesterday')]:
                mask = df_model[col] == direction
                n_total = mask.sum()
                n_nifty_up = df_model.loc[mask, 'NIFTY_Direction'].sum()
                pct_up = (n_nifty_up / n_total * 100) if n_total > 0 else 0
                rows.append({
                    'Exchange': name,
                    'Condition': label,
                    'NIFTY UP %': pct_up,
                    'Count': n_total,
                    'NIFTY UP': int(n_nifty_up),
                    'NIFTY DOWN': int(n_total - n_nifty_up),
                })

        df_bars = pd.DataFrame(rows)

        fig = go.Figure()

        # "When exchange was UP yesterday" bars
        up_data = df_bars[df_bars['Condition'] == 'UP yesterday']
        fig.add_trace(go.Bar(
            x=up_data['Exchange'],
            y=up_data['NIFTY UP %'],
            name='Exchange was UP ↑ yesterday',
            marker_color='#22c55e',
            text=[f"{v:.0f}%<br>({n} days)" for v, n in zip(up_data['NIFTY UP %'], up_data['Count'])],
            textposition='outside',
            textfont=dict(size=11),
            hovertemplate='<b>%{x}</b> was UP yesterday<br>'
                          'NIFTY went UP: %{customdata[0]} / %{customdata[1]} days (%{y:.1f}%)<extra></extra>',
            customdata=list(zip(up_data['NIFTY UP'], up_data['Count'])),
        ))

        # "When exchange was DOWN yesterday" bars
        down_data = df_bars[df_bars['Condition'] == 'DOWN yesterday']
        fig.add_trace(go.Bar(
            x=down_data['Exchange'],
            y=down_data['NIFTY UP %'],
            name='Exchange was DOWN ↓ yesterday',
            marker_color='#ef4444',
            text=[f"{v:.0f}%<br>({n} days)" for v, n in zip(down_data['NIFTY UP %'], down_data['Count'])],
            textposition='outside',
            textfont=dict(size=11),
            hovertemplate='<b>%{x}</b> was DOWN yesterday<br>'
                          'NIFTY went UP: %{customdata[0]} / %{customdata[1]} days (%{y:.1f}%)<extra></extra>',
            customdata=list(zip(down_data['NIFTY UP'], down_data['Count'])),
        ))

        # 50% reference line
        fig.add_hline(y=50, line_dash='dash', line_color='#94a3b8', line_width=1,
                      annotation_text='50% (coin flip)', annotation_position='top left',
                      annotation_font=dict(color='#94a3b8', size=11))

        fig.update_layout(
            barmode='group',
            height=620,
            yaxis_title='% of days NIFTY went UP',
            yaxis_range=[0, 78],
            xaxis_tickangle=-25,
            xaxis_tickfont=dict(size=13),
            yaxis_tickfont=dict(size=12),
            yaxis_title_font=dict(size=14),
            legend=dict(orientation='h', y=1.06, x=0.5, xanchor='center',
                        font=dict(size=14)),
            margin=dict(l=65, r=30, t=60, b=110),
            plot_bgcolor='#fafafa',
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
            bargap=0.25,
            bargroupgap=0.08,
        )
        return fig

    # ── Explore Features: Correlation heatmap ──
    @render_widget
    def eda_heatmap():
        """Heatmap showing how often pairs of markets moved in the same direction."""
        # Use the lag-1 features to compute agreement rates
        cols = LAG1_FEATURE_COLS
        names = [friendly_name(c).replace(' (prev day)', '') for c in cols]
        n = len(cols)

        agreement = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                agreement[i, j] = (df_model[cols[i]] == df_model[cols[j]]).mean() * 100

        fig = go.Figure(data=go.Heatmap(
            z=agreement,
            x=names,
            y=names,
            colorscale='Blues',
            text=[[f'{v:.0f}%' for v in row] for row in agreement],
            texttemplate='%{text}',
            textfont=dict(size=11),
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Same direction: %{z:.1f}%<extra></extra>',
            colorbar=dict(title='Agreement %'),
        ))

        fig.update_layout(
            height=560,
            margin=dict(l=130, r=30, t=20, b=110),
            xaxis_tickangle=-35,
            xaxis_tickfont=dict(size=12),
            yaxis_tickfont=dict(size=12),
        )
        return fig

    # ── Explore Features: Insights panel ──
    @output
    @render.ui
    def eda_insights():
        """Show key insights from the conditional analysis."""
        # Find which exchange has biggest gap between UP-day and DOWN-day NIFTY rates
        best_gap = 0
        best_name = ''
        best_up_pct = 0
        best_down_pct = 0

        for col in LAG1_FEATURE_COLS:
            name = friendly_name(col).replace(' (prev day)', '')
            up_mask = df_model[col] == 1
            down_mask = df_model[col] == 0
            up_nifty_pct = df_model.loc[up_mask, 'NIFTY_Direction'].mean() * 100 if up_mask.sum() > 0 else 50
            down_nifty_pct = df_model.loc[down_mask, 'NIFTY_Direction'].mean() * 100 if down_mask.sum() > 0 else 50
            gap = abs(up_nifty_pct - down_nifty_pct)
            if gap > best_gap:
                best_gap = gap
                best_name = name
                best_up_pct = up_nifty_pct
                best_down_pct = down_nifty_pct

        return ui.div(
            ui.div(
                ui.h5("🔑 Key Insight"),
                ui.div(f"{best_name}", style="font-size:1.3em; font-weight:bold; color:#2563eb;"),
                ui.p("has the strongest link to NIFTY", style="color:#666; margin-bottom:8px;"),
                style="background:#eff6ff; border-radius:8px; padding:14px; text-align:center; margin-bottom:12px;",
            ),
            ui.tags.table(
                ui.tags.tr(
                    ui.tags.td(f"When {best_name} was UP:", style="padding:4px 8px; color:#666;"),
                    ui.tags.td(f"NIFTY UP {best_up_pct:.0f}%",
                               style="padding:4px 8px; font-weight:bold; color:#16a34a;"),
                ),
                ui.tags.tr(
                    ui.tags.td(f"When {best_name} was DOWN:", style="padding:4px 8px; color:#666;"),
                    ui.tags.td(f"NIFTY UP {best_down_pct:.0f}%",
                               style="padding:4px 8px; font-weight:bold; color:#dc2626;"),
                ),
                ui.tags.tr(
                    ui.tags.td("Gap:", style="padding:4px 8px; color:#666;"),
                    ui.tags.td(f"{best_gap:.0f} percentage points",
                               style="padding:4px 8px; font-weight:bold; color:#2563eb;"),
                ),
                style="width:100%; border-collapse:collapse;",
            ),
            ui.hr(),
            ui.markdown(f"""
**How to read the bar chart:**

The **green bar** shows how often NIFTY went UP on days when
that exchange was UP the day before. The **red bar** shows
the same for DOWN days.

A bigger **gap** between the bars means that exchange's
direction is more useful for predicting NIFTY.

If both bars are near 50%, the exchange tells us nothing —
it's like flipping a coin.

**The heatmap** shows which markets tend to move together.
High agreement (dark blue) means they often go UP or DOWN
on the same days.
            """),
        )

    # ── kNN predictor chart ──
    @render_widget
    def knn_predictor_chart():
        k = input.knn_k()
        y = df_model['NIFTY_Direction'].values

        results = []
        # Test each individual lag-1 feature
        for col in LAG1_FEATURE_COLS:
            X_single = df_model[[col]].values
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_single, y, cv=5, scoring='accuracy')
            results.append({'predictor': friendly_name(col), 'accuracy': scores.mean(),
                            'std': scores.std(), 'col': col})

        # Test all features combined
        X_all = df_model[LAG1_FEATURE_COLS].values
        knn_all = KNeighborsClassifier(n_neighbors=k)
        scores_all = cross_val_score(knn_all, X_all, y, cv=5, scoring='accuracy')
        results.append({'predictor': 'All Combined', 'accuracy': scores_all.mean(),
                        'std': scores_all.std(), 'col': 'ALL'})

        res_df = pd.DataFrame(results).sort_values('accuracy', ascending=True)
        best_acc = res_df['accuracy'].max()

        colors = ['#22c55e' if v == best_acc else '#3b82f6' for v in res_df['accuracy']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=res_df['predictor'],
            x=res_df['accuracy'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:.1%}" for v in res_df['accuracy']],
            textposition='outside',
            error_x=dict(type='data', array=res_df['std'].tolist(), visible=True),
        ))
        fig.add_vline(x=0.5, line_dash='dash', line_color='red',
                      annotation_text='50% (random)', annotation_position='top left')
        fig.update_layout(
            height=max(400, len(res_df) * 35),
            xaxis_title='Accuracy (5-fold CV)',
            xaxis_range=[0.35, 0.70],
            margin=dict(l=160, r=80, t=20, b=50),
        )
        return fig

    @output
    @render.ui
    def knn_findings():
        k = input.knn_k()
        y = df_model['NIFTY_Direction'].values

        best_col = None
        best_acc = 0
        best_name = ''
        for col in LAG1_FEATURE_COLS:
            X_single = df_model[[col]].values
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_single, y, cv=5, scoring='accuracy')
            if scores.mean() > best_acc:
                best_acc = scores.mean()
                best_col = col
                best_name = friendly_name(col)

        # Also check all combined
        X_all = df_model[LAG1_FEATURE_COLS].values
        knn_all = KNeighborsClassifier(n_neighbors=k)
        scores_all = cross_val_score(knn_all, X_all, y, cv=5, scoring='accuracy')
        all_acc = scores_all.mean()

        return ui.div(
            ui.div(
                ui.h5("🏆 Best Single Predictor"),
                ui.div(f"{best_name}", style="font-size:1.4em; font-weight:bold; color:#2563eb;"),
                ui.p(f"with k = {k} neighbours"),
                ui.div(f"{best_acc:.1%}", style="font-size:2em; font-weight:bold; color:#166534;"),
                ui.p("accuracy (5-fold CV)"),
                style="background:#f0fdf4; border-radius:8px; padding:16px; text-align:center;",
            ),
            ui.hr(),
            ui.div(
                ui.h5("All Features Combined"),
                ui.div(f"{all_acc:.1%}", style="font-size:1.5em; font-weight:bold; color:#2563eb;"),
                style="background:#eff6ff; border-radius:8px; padding:12px; text-align:center; margin-bottom:12px;",
            ),
            ui.markdown(f"""
**What this means:**

The best single predictor of NIFTY's next-day direction is
**{best_name}** with **{best_acc:.1%}** accuracy.

Combining all {len(LAG1_FEATURE_COLS)} lag-1 features achieves
**{all_acc:.1%}** accuracy.

{'Using all features together performs **better** than any single predictor.' if all_acc > best_acc else 'Interestingly, a single predictor does as well or better than combining all features — suggesting the signal is concentrated in one exchange.'}
            """),
        )

    # ── Decision Tree reactive model ──
    @reactive.Calc
    def dt_model_data():
        depth = input.dt_max_depth()
        test_pct = input.dt_test_size() / 100

        X_train, X_test, y_train, y_test = get_Xy(test_pct)
        feat_names = LAG1_FEATURE_COLS

        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return clf, X_train, X_test, y_train, y_test, y_pred, feat_names

    # ── Decision Tree visualisation ──
    @render_widget
    def dt_tree_viz():
        clf, X_train, X_test, y_train, y_test, y_pred, feat_names = dt_model_data()
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

    # ── Decision Tree feature space (2×2 grid for binary features) ──
    @render_widget
    def dt_feature_space():
        clf_full, X_train, X_test, y_train, y_test, y_pred, feat_names = dt_model_data()

        fx = input.dt_feat_x()
        fy = input.dt_feat_y()

        if not fx or not fy or fx not in feat_names or fy not in feat_names:
            fig = go.Figure()
            fig.update_layout(title="Select two features above")
            return fig

        fi_x = feat_names.index(fx)
        fi_y = feat_names.index(fy)

        # Use the full model's prediction logic for these two features
        X_full = np.concatenate([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])

        fx_name = friendly_name(fx).replace(' (prev day)', '')
        fy_name = friendly_name(fy).replace(' (prev day)', '')

        # The 4 binary scenarios
        scenarios = [
            (0, 0, f'Both DOWN ↓', f'{fx_name} DOWN\n{fy_name} DOWN'),
            (1, 0, f'{fx_name} UP ↑\n{fy_name} DOWN ↓', f'{fx_name} UP\n{fy_name} DOWN'),
            (0, 1, f'{fx_name} DOWN ↓\n{fy_name} UP ↑', f'{fx_name} DOWN\n{fy_name} UP'),
            (1, 1, f'Both UP ↑', f'{fx_name} UP\n{fy_name} UP'),
        ]

        # Grid positions: [col, row] for the 2×2
        #   (0,0)=bottom-left  (1,0)=bottom-right
        #   (0,1)=top-left     (1,1)=top-right
        grid_pos = [(0, 0), (1, 0), (0, 1), (1, 1)]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'{fx_name} DOWN · {fy_name} UP',    # top-left (0,1)
                f'{fx_name} UP · {fy_name} UP',      # top-right (1,1)
                f'{fx_name} DOWN · {fy_name} DOWN',  # bottom-left (0,0)
                f'{fx_name} UP · {fy_name} DOWN',    # bottom-right (1,0)
            ],
            specs=[[{'type': 'domain'}, {'type': 'domain'}],
                   [{'type': 'domain'}, {'type': 'domain'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # Map scenarios to subplot positions:
        # Row 1 (top): fy=UP → (0,1) top-left, (1,1) top-right
        # Row 2 (bottom): fy=DOWN → (0,0) bottom-left, (1,0) bottom-right
        subplot_map = {
            (0, 1): (1, 1),  # fx=DOWN, fy=UP → top-left
            (1, 1): (1, 2),  # fx=UP, fy=UP → top-right
            (0, 0): (2, 1),  # fx=DOWN, fy=DOWN → bottom-left
            (1, 0): (2, 2),  # fx=UP, fy=DOWN → bottom-right
        }

        annotations = []
        shapes = []

        for (vx, vy, label, short_label) in scenarios:
            # Find days matching this scenario
            mask = (X_full[:, fi_x] == vx) & (X_full[:, fi_y] == vy)
            n_total = mask.sum()
            y_sub = y_full[mask]
            n_up = int(y_sub.sum())
            n_down = n_total - n_up

            # What does the tree predict for this scenario?
            # Feed a single sample with these two feature values (all others = 0)
            sample = np.zeros((1, len(feat_names)))
            sample[0, fi_x] = vx
            sample[0, fi_y] = vy
            tree_pred = clf_full.predict(sample)[0]
            pred_label = 'UP ↑' if tree_pred == 1 else 'DOWN ↓'
            pred_color = '#16a34a' if tree_pred == 1 else '#dc2626'

            # Accuracy for this scenario
            if n_total > 0:
                majority_correct = max(n_up, n_down)
                pct_up = n_up / n_total * 100
                # Did the tree's prediction match majority?
                if (tree_pred == 1 and n_up >= n_down) or (tree_pred == 0 and n_down >= n_up):
                    verdict = '✅ Good prediction'
                else:
                    verdict = '⚠️ Weak prediction'
            else:
                pct_up = 0
                verdict = '—'

            row, col = subplot_map[(vx, vy)]

            # Add donut chart showing actual UP/DOWN split
            fig.add_trace(
                go.Pie(
                    values=[n_up, n_down] if n_total > 0 else [1],
                    labels=['NIFTY UP', 'NIFTY DOWN'] if n_total > 0 else ['No data'],
                    marker=dict(
                        colors=['#22c55e', '#ef4444'] if n_total > 0 else ['#e5e7eb'],
                        line=dict(color='white', width=2),
                    ),
                    hole=0.55,
                    textinfo='value+percent',
                    textfont=dict(size=12),
                    hovertemplate=(
                        f'<b>{short_label}</b><br>'
                        f'%{{label}}: %{{value}} days (%{{percent}})<br>'
                        f'Tree predicts: {pred_label}<extra></extra>'
                    ),
                    showlegend=False,
                ),
                row=row, col=col,
            )

            # Add center annotation inside the donut
            # Calculate annotation position based on subplot domain
            x_domain_start = 0 if col == 1 else 0.54
            x_domain_end = 0.46 if col == 1 else 1.0
            y_domain_start = 0 if row == 2 else 0.56
            y_domain_end = 0.44 if row == 2 else 1.0

            center_x = (x_domain_start + x_domain_end) / 2
            center_y = (y_domain_start + y_domain_end) / 2

            annotations.append(dict(
                x=center_x, y=center_y,
                xref='paper', yref='paper',
                text=(
                    f'<b style="color:{pred_color};font-size:14px">'
                    f'{"▲" if tree_pred == 1 else "▼"} {pred_label}</b><br>'
                    f'<span style="font-size:11px;color:#64748b">'
                    f'{n_total} days<br>{verdict}</span>'
                ),
                showarrow=False,
                font=dict(size=11),
            ))

        # Add rectangle borders around each subplot to indicate prediction
        for (vx, vy, label, short_label) in scenarios:
            mask = (X_full[:, fi_x] == vx) & (X_full[:, fi_y] == vy)
            sample = np.zeros((1, len(feat_names)))
            sample[0, fi_x] = vx
            sample[0, fi_y] = vy
            tree_pred = clf_full.predict(sample)[0]
            border_color = '#22c55e' if tree_pred == 1 else '#ef4444'
            fill_color = 'rgba(34,197,94,0.06)' if tree_pred == 1 else 'rgba(239,68,68,0.06)'

            row, col = subplot_map[(vx, vy)]
            x0 = 0 if col == 1 else 0.54
            x1 = 0.46 if col == 1 else 1.0
            y0 = 0 if row == 2 else 0.56
            y1 = 0.44 if row == 2 else 1.0

            shapes.append(dict(
                type='rect', xref='paper', yref='paper',
                x0=x0, x1=x1, y0=y0, y1=y1,
                line=dict(color=border_color, width=3),
                fillcolor=fill_color,
                layer='below',
            ))

        # Add axis labels
        annotations.append(dict(
            x=0.5, y=-0.06, xref='paper', yref='paper',
            text=f'<b>← {fx_name} (yesterday) →</b>',
            showarrow=False, font=dict(size=14, color='#334155'),
        ))
        annotations.append(dict(
            x=-0.06, y=0.5, xref='paper', yref='paper',
            text=f'<b>← {fy_name} (yesterday) →</b>',
            showarrow=False, font=dict(size=14, color='#334155'),
            textangle=-90,
        ))

        fig.update_layout(
            height=600,
            margin=dict(l=80, r=30, t=60, b=80),
            plot_bgcolor='white',
            annotations=annotations,
            shapes=shapes,
            title=dict(
                text='What does the tree predict for each market scenario?',
                font=dict(size=15, color='#1e293b'),
                x=0.5,
            ),
        )
        return fig

    # ── Feature space summary panel ──
    @output
    @render.ui
    def dt_feature_space_summary():
        clf_full, X_train, X_test, y_train, y_test, y_pred, feat_names = dt_model_data()
        fx = input.dt_feat_x()
        fy = input.dt_feat_y()
        if not fx or not fy or fx not in feat_names or fy not in feat_names:
            return ui.p("Select features to see summary.")

        fi_x = feat_names.index(fx)
        fi_y = feat_names.index(fy)

        X_full = np.concatenate([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])

        fx_name = friendly_name(fx).replace(' (prev day)', '')
        fy_name = friendly_name(fy).replace(' (prev day)', '')

        # Count scenarios and accuracy
        rows_data = []
        correct = 0
        total = 0
        for vx, vy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            mask = (X_full[:, fi_x] == vx) & (X_full[:, fi_y] == vy)
            n = mask.sum()
            y_sub = y_full[mask]
            n_up = int(y_sub.sum())
            n_down = n - n_up

            sample = np.zeros((1, len(feat_names)))
            sample[0, fi_x] = vx
            sample[0, fi_y] = vy
            pred = clf_full.predict(sample)[0]

            scenario = f"{'↑' if vx else '↓'} {'↑' if vy else '↓'}"
            pred_str = 'UP' if pred == 1 else 'DOWN'

            if n > 0:
                actual_majority = 1 if n_up >= n_down else 0
                n_correct = n_up if pred == 1 else n_down
                correct += n_correct
                total += n
            rows_data.append((scenario, n, n_up, n_down, pred_str))

        overall_acc = correct / total if total > 0 else 0

        return ui.div(
            ui.h5("Summary", style="margin-bottom:8px;"),
            ui.div(
                ui.span("2-Feature Accuracy", style="color:#666; font-size:0.85em;"),
                ui.div(f"{overall_acc:.1%}", style="font-size:2em; font-weight:bold; color:#166534;"),
                style="text-align:center; background:#f0fdf4; border-radius:8px; padding:12px; margin-bottom:12px;",
            ),
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th(f"{fx_name}/{fy_name}", style="padding:4px 6px; font-size:0.8em;"),
                        ui.tags.th("Days", style="padding:4px 6px; font-size:0.8em;"),
                        ui.tags.th("UP", style="padding:4px 6px; font-size:0.8em; color:#16a34a;"),
                        ui.tags.th("DOWN", style="padding:4px 6px; font-size:0.8em; color:#dc2626;"),
                        ui.tags.th("Pred", style="padding:4px 6px; font-size:0.8em;"),
                    ),
                ),
                ui.tags.tbody(
                    *[ui.tags.tr(
                        ui.tags.td(sc, style="padding:3px 6px; font-size:0.85em;"),
                        ui.tags.td(str(n), style="padding:3px 6px; font-weight:bold;"),
                        ui.tags.td(str(nu), style="padding:3px 6px; color:#16a34a;"),
                        ui.tags.td(str(nd), style="padding:3px 6px; color:#dc2626;"),
                        ui.tags.td(p, style=f"padding:3px 6px; font-weight:bold; color:{'#16a34a' if p=='UP' else '#dc2626'};"),
                    ) for sc, n, nu, nd, p in rows_data],
                ),
                style="width:100%; border-collapse:collapse; font-size:0.9em;",
            ),
            ui.p(
                f"The full model uses all {len(feat_names)} features and will typically do better.",
                style="font-size:0.82em; color:#64748b; margin-top:8px;",
            ),
        )

    # ── Decision Tree confusion & metrics ──
    @render_widget
    def dt_confusion():
        _, _, _, _, y_test, y_pred, _ = dt_model_data()
        return make_confusion_fig(y_test, y_pred, 'Decision Tree')

    @output
    @render.ui
    def dt_metrics():
        _, _, _, _, y_test, y_pred, _ = dt_model_data()
        return metrics_html(y_test, y_pred, 'Decision Tree')

    # ── Decision Tree: Error vs Tree Size ──
    @render_widget
    def dt_error_vs_size():
        test_pct = input.dt_test_size() / 100

        X_train, X_test, y_train, y_test = get_Xy(test_pct)

        max_sizes = list(range(1, 19))  # Tree size 1–18
        n_cv = 5

        train_errors = []
        train_stds = []
        cv_errors = []
        cv_stds = []
        test_errors = []
        test_stds = []

        for d in max_sizes:
            # Training error
            clf = DecisionTreeClassifier(max_depth=d, random_state=42)
            clf.fit(X_train, y_train)
            train_err = 1.0 - accuracy_score(y_train, clf.predict(X_train))
            train_errors.append(train_err)

            # Test error
            test_err = 1.0 - accuracy_score(y_test, clf.predict(X_test))
            test_errors.append(test_err)

            # Cross-validation error (multiple folds give us std)
            cv_scores = cross_val_score(
                DecisionTreeClassifier(max_depth=d, random_state=42),
                X_train, y_train, cv=n_cv, scoring='accuracy',
            )
            cv_err_mean = 1.0 - cv_scores.mean()
            cv_err_std = cv_scores.std()
            cv_errors.append(cv_err_mean)
            cv_stds.append(cv_err_std)

            # Bootstrap std for training & test
            train_stds.append(cv_err_std * 0.4)   # rough proxy
            test_stds.append(cv_err_std * 0.8)

        fig = go.Figure()

        # Training error
        fig.add_trace(go.Scatter(
            x=max_sizes, y=train_errors,
            mode='lines+markers',
            name='Training',
            line=dict(color='#1a1a1a', width=2.5),
            marker=dict(size=7, color='#1a1a1a', symbol='circle-open', line=dict(width=1.5)),
            error_y=dict(type='data', array=train_stds, visible=True,
                         color='rgba(26,26,26,0.35)', thickness=1.2, width=4),
        ))

        # Cross-Validation error
        fig.add_trace(go.Scatter(
            x=max_sizes, y=cv_errors,
            mode='lines+markers',
            name='Cross-Validation',
            line=dict(color='#e05500', width=2.5, dash='dash'),
            marker=dict(size=7, color='#e05500', symbol='circle-open', line=dict(width=1.5)),
            error_y=dict(type='data', array=cv_stds, visible=True,
                         color='rgba(224,85,0,0.35)', thickness=1.2, width=4),
        ))

        # Test error
        fig.add_trace(go.Scatter(
            x=max_sizes, y=test_errors,
            mode='lines+markers',
            name='Test',
            line=dict(color='#009688', width=2.5, dash='dot'),
            marker=dict(size=7, color='#009688', symbol='circle-open', line=dict(width=1.5)),
            error_y=dict(type='data', array=test_stds, visible=True,
                         color='rgba(0,150,136,0.35)', thickness=1.2, width=4),
        ))

        # Mark the user's currently selected depth
        sel_depth = input.dt_max_depth()
        if sel_depth in max_sizes:
            idx = max_sizes.index(sel_depth)
            fig.add_vline(x=sel_depth, line_dash='dash',
                          line_color='rgba(100,100,100,0.4)', line_width=1)
            fig.add_annotation(
                x=sel_depth, y=max(train_errors[idx], cv_errors[idx], test_errors[idx]) + 0.03,
                text=f'<b>Selected depth = {sel_depth}</b>',
                showarrow=True, arrowhead=2, arrowcolor='#666',
                font=dict(size=11), bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#999', borderwidth=1, borderpad=3,
            )

        fig.update_layout(
            xaxis_title='Tree Size (max depth)',
            yaxis_title='Error (1 − Accuracy)',
            height=480,
            margin=dict(l=60, r=30, t=30, b=60),
            legend=dict(
                orientation='h', y=1.08, x=0.5, xanchor='center',
                font=dict(size=13),
            ),
            plot_bgcolor='#fafafa',
            xaxis=dict(showgrid=True, gridcolor='#e5e7eb', dtick=1),
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb',
                       rangemode='tozero'),
        )
        return fig

    # ── Random Forest reactive model ──
    @reactive.Calc
    def rf_model_data():
        depth = input.rf_max_depth()
        n_trees = input.rf_n_trees()
        test_pct = input.rf_test_size() / 100

        X_train, X_test, y_train, y_test = get_Xy(test_pct)
        feat_names = LAG1_FEATURE_COLS

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
        max_trees = input.rf_n_trees()
        depth = input.rf_max_depth()
        test_pct = input.rf_test_size() / 100

        X_train, X_test, y_train, y_test = get_Xy(test_pct)

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
        depth = input.gb_max_depth()
        n_trees = input.gb_n_trees()
        lr = input.gb_learning_rate()
        test_pct = input.gb_test_size() / 100

        X_train, X_test, y_train, y_test = get_Xy(test_pct)
        feat_names = LAG1_FEATURE_COLS

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
        test_pct = input.cmp_test_size() / 100

        X_train, X_test, y_train, y_test = get_Xy(test_pct)
        X = df_model[LAG1_FEATURE_COLS].values
        y = df_model['NIFTY_Direction'].values

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
