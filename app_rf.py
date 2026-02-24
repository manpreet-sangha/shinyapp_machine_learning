"""
app_rf.py — Tab 4: Random Forest
──────────────────────────────────
UI panel and server functions for the Random Forest tab.
"""

import math
import numpy as np
import plotly.graph_objects as go

from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from app_data import (
    get_Xy, LAG1_FEATURE_COLS, friendly_name,
    make_confusion_fig, metrics_html, make_roc_fig,
)


# ═══════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════

def rf_ui():
    return ui.nav_panel(
        ui.span(ui.tags.i(class_="bi bi-trees me-1"), "Random Forest"),
        ui.layout_columns(
            ui.input_slider("rf_n_trees", "Number of trees:", min=10, max=300, value=100, step=10),
            ui.input_slider("rf_max_depth", "Max tree depth:", min=1, max=10, value=4, step=1),
            ui.input_slider("rf_test_size", "Test set size (%):", min=10, max=40, value=20, step=5),
            col_widths=[4, 4, 4],
        ),
        ui.navset_card_tab(
            ui.nav_panel(
                "Understanding Random Forests",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("What is a Random Forest?"),
                        ui.markdown("""
Instead of relying on **one** tree, a **Random Forest** builds
**many trees** (a "forest") and lets them **vote** on the
prediction.

Each tree sees a random subset of the data and features, so
they each learn slightly different patterns. The **majority
vote** is usually more accurate and stable than any single tree.

---

**How does Bagging work?**

Random Forest is powered by **Bootstrap Aggregation (Bagging)**:

1. **Resample** -- Take repeated random samples *with
   replacement* from the training data to create **B**
   different bootstrapped training datasets (each the same
   size as the original, but with some rows repeated and
   others left out).

2. **Train** -- Build **B** separate classification trees,
   one on each bootstrapped dataset. Because each tree sees
   different data, they each pick up on different patterns.

3. **Predict** -- For a new day, run it through all **B**
   trees and collect **B** individual predictions.

4. **Majority vote** -- The final prediction is whichever
   direction (UP or DOWN) the most trees agree on.

Use **B sufficiently large** (e.g. 100+) so the vote is stable.
                        """),
                    ),
                    ui.card(
                        ui.card_header("Out-of-Bag (OOB) & Limiting Clues"),
                        ui.markdown("""
**What about Out-of-Bag (OOB)?**

Each bootstrap sample leaves out roughly **one-third** of the
training rows. These left-out rows are called **Out-of-Bag
(OOB)** samples.

We can test each tree on its own OOB data and average the
results to get an **OOB error estimate** — a built-in
accuracy check that does not need a separate test set.

Check the **Trees vs Error** tab to see how OOB error compares
to the test error as the forest grows.

---

**How many clues should each tree see?**

At every branch, a tree needs to pick the best market to
split on. We can choose **how many markets (clues)** each
tree is allowed to consider at that moment:

- **All 10 markets** — every tree sees everything. This
  sounds great, but all trees end up making the same
  decision because they all latch onto the most obvious
  market. This is just Bagging.
- **5 markets** — each tree sees a random half. Trees
  start to differ from one another.
- **3 markets** (the standard) — each tree sees very few
  options, so every tree is forced to learn something
  unique. The group's combined answer is most reliable.

**Why does limiting clues help?** Global stock markets often
move together. If every tree can see all of them, they all
copy the same dominant signal. Restricting clues forces each
tree to explore **different** markets — their combined vote
becomes more **diverse and accurate**.

Check the **How Many Clues Per Split?** tab to compare these
settings on our data.
                        """),
                    ),
                    col_widths=[6, 6],
                ),
            ),
            ui.nav_panel(
                "Feature Importance",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Variable Importance (scaled 0 -- 100)"),
                        output_widget("rf_importance"),
                    ),
                    ui.card(
                        ui.card_header("How is importance measured?"),
                        ui.markdown("""
When the forest builds its trees, each split uses one
feature to separate UP and DOWN days. The **Gini index**
measures how well a split separates the two classes.

**Variable importance** adds up how much each feature
reduces the Gini index across *all* splits in *all* trees
in the forest:

- A feature that appears in many splits and produces
  large Gini reductions is **highly important**.
- A feature that rarely helps separate UP from DOWN
  has **low importance**.

The values are **scaled from 0 to 100** so the most
important feature = 100 and the rest are relative to it.

**In short:** taller bars = that market's previous-day
direction is more useful for predicting NIFTY's
next-day move.
                        """),
                    ),
                    col_widths=[7, 5],
                ),
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
                "ROC Curve",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("ROC Curve & AUC"),
                        output_widget("rf_roc_curve"),
                    ),
                    ui.card(
                        ui.card_header("What is a ROC Curve?"),
                        ui.markdown("""
**ROC** stands for *Receiver Operating Characteristic*.

Think of it this way: your model has a dial that controls
how aggressively it predicts "UP". As you turn the dial:

- **Turn it up** -- it catches more real UP days
  (higher Sensitivity) but also makes more false alarms
  (lower Specificity)
- **Turn it down** -- fewer false alarms, but it misses
  more real UP days

The **ROC curve** plots this trade-off at every possible
dial setting.

**AUC** (Area Under the Curve) summarises the whole curve
into a single number:

- **AUC = 1.0** -- perfect model
- **AUC = 0.5** -- no better than flipping a coin
  (the red dashed line)
- **AUC > 0.5** -- the model has *some* predictive power;
  the further above the red line, the better

*In short: the more the blue curve bows toward the
top-left corner, the better the model is at
distinguishing UP days from DOWN days.*
                        """),
                    ),
                    col_widths=[7, 5],
                ),
            ),
            ui.nav_panel(
                "Trees vs Error",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Test Error & OOB Error vs Number of Trees"),
                        output_widget("rf_learning_curve"),
                    ),
                    ui.card(
                        ui.card_header("Reading this chart"),
                        ui.markdown("""
**What is happening here?**

As we add more trees to the forest, both error measures
tend to **drop and stabilise**:

- **Test Error** (dark line) -- Measured on a held-out
  portion of the data the model never trained on.
- **OOB Error** (teal line) -- Each tree is tested on the
  rows it did *not* see during its bootstrap sample, then
  the results are averaged.

**Key take-aways:**

- **Both lines decrease** as the number of trees grows,
  showing the ensemble is improving.
- After a certain point the curves **flatten** -- adding
  more trees no longer helps much but does not hurt either.
- **OOB error tracks test error closely**, confirming that
  bagging's built-in validation works well.
- A stable OOB line means the forest has reached a
  **good size** and we are not overfitting.
                        """),
                    ),
                    col_widths=[7, 5],
                ),
            ),
            ui.nav_panel(
                "How Many Clues Per Split?",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Test Error vs Number of Trees — Varying the Clue Limit"),
                        output_widget("rf_max_features_chart"),
                    ),
                    ui.card(
                        ui.card_header("What does this chart show?"),
                        ui.markdown("""
Imagine each tree in the forest has to make a decision at
every branch. Before deciding, it looks at a handful of
**clues** (market directions from yesterday). This chart
tests what happens when we change **how many clues** each
tree is allowed to consider:

- **All 10 clues** (orange) — every tree sees every market.
  This sounds ideal but it means all trees tend to make the
  same decision, so there is less benefit from having many
  trees. This is called **Bagging**.

- **5 clues** (blue) — each tree only sees half the markets
  at random. Trees start making different decisions, which
  helps the group catch patterns that one tree might miss.

- **3 clues** (teal) — this is the standard **Random Forest**
  setting. Each tree sees very few markets, so every tree
  is forced to learn something unique. The group's combined
  answer is usually the most reliable.

**Why does limiting clues help?** Global stock markets often
move together. If every tree sees all markets, they all
latch onto the same dominant market and ignore the rest.
By restricting clues, we force trees to explore different
markets — their combined vote becomes more **diverse and
stable**, which typically means a **lower error rate**.

*The line that sits lowest on the chart is the best
setting for our data.*
                        """),
                    ),
                    col_widths=[7, 5],
                ),
            ),
        ),
    )


# ═══════════════════════════════════════════════════════════
#  Server
# ═══════════════════════════════════════════════════════════

def rf_server(input, output, session):

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
        max_imp = importances.max()
        scaled = (importances / max_imp * 100) if max_imp > 0 else importances
        indices = np.argsort(scaled)

        names = [friendly_name(feat_names[i]) for i in indices]
        vals = scaled[indices]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=names, x=vals, orientation='h',
            marker_color='#dc2626',
            text=[f'{v:.0f}' for v in vals],
            textposition='outside', textfont=dict(size=11),
        ))
        fig.update_layout(
            height=500, xaxis_title='Variable Importance',
            xaxis_range=[0, 110],
            margin=dict(l=200, r=40, t=20, b=50),
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
    def rf_roc_curve():
        clf, _, X_test, _, y_test, _, _ = rf_model_data()
        y_prob = clf.predict_proba(X_test)[:, 1]
        return make_roc_fig(y_test, y_prob, 'Random Forest')

    @render_widget
    def rf_learning_curve():
        max_trees = input.rf_n_trees()
        depth = input.rf_max_depth()
        test_pct = input.rf_test_size() / 100

        X_train, X_test, y_train, y_test = get_Xy(test_pct)

        tree_counts = list(range(10, max_trees + 1, 10))
        test_errors = []
        oob_errors = []
        for n in tree_counts:
            rf = RandomForestClassifier(
                n_estimators=n, max_depth=depth,
                random_state=42, n_jobs=-1, oob_score=True,
            )
            rf.fit(X_train, y_train)
            test_errors.append(1 - accuracy_score(y_test, rf.predict(X_test)))
            oob_errors.append(1 - rf.oob_score_)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tree_counts, y=test_errors, mode='lines',
            name='Test Error', line=dict(color='#1e293b', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=tree_counts, y=oob_errors, mode='lines',
            name='OOB Error', line=dict(color='#14b8a6', width=2),
        ))
        fig.add_hline(y=0.5, line_dash='dash', line_color='gray',
                      annotation_text='50% (coin flip)')
        fig.update_layout(
            height=450,
            xaxis_title='Number of Trees', yaxis_title='Error',
            yaxis_range=[0, max(max(test_errors), max(oob_errors)) * 1.15],
            legend=dict(orientation='h', y=-0.15),
            margin=dict(l=50, r=20, t=20, b=60),
        )
        return fig

    @render_widget
    def rf_max_features_chart():
        max_trees = input.rf_n_trees()
        depth = input.rf_max_depth()
        test_pct = input.rf_test_size() / 100

        X_train, X_test, y_train, y_test = get_Xy(test_pct)
        p = X_train.shape[1]

        m_settings = {
            f'All {p} clues (Bagging)': p,
            f'Half ({p // 2} clues)': p // 2,
            f'Square root ({int(math.sqrt(p))} clues — RF default)': int(math.sqrt(p)),
        }
        colours = {
            f'All {p} clues (Bagging)': '#f59e0b',
            f'Half ({p // 2} clues)': '#3b82f6',
            f'Square root ({int(math.sqrt(p))} clues — RF default)': '#14b8a6',
        }

        tree_counts = list(range(10, max_trees + 1, 10))
        fig = go.Figure()

        for label, m_val in m_settings.items():
            errors = []
            for n in tree_counts:
                rf = RandomForestClassifier(
                    n_estimators=n, max_depth=depth,
                    max_features=m_val, random_state=42, n_jobs=-1,
                )
                rf.fit(X_train, y_train)
                errors.append(1 - accuracy_score(y_test, rf.predict(X_test)))
            fig.add_trace(go.Scatter(
                x=tree_counts, y=errors, mode='lines',
                name=label, line=dict(color=colours[label], width=2),
            ))

        fig.update_layout(
            height=450,
            xaxis_title='Number of Trees',
            yaxis_title='Test Classification Error',
            legend=dict(orientation='h', y=-0.15),
            margin=dict(l=50, r=20, t=20, b=60),
        )
        return fig
