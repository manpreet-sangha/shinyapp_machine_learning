"""
app_gb.py — Tab 5: Gradient Boosting
──────────────────────────────────────
UI panel and server functions for the Gradient Boosting tab.
"""

import numpy as np
import plotly.graph_objects as go

from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from app_data import (
    get_Xy, LAG1_FEATURE_COLS, friendly_name,
    make_confusion_fig, metrics_html, make_roc_fig,
)


# ═══════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════

def gb_ui():
    return ui.nav_panel(
        ui.span(ui.tags.i(class_="bi bi-graph-up-arrow me-1"), "Gradient Boosting"),
        ui.layout_columns(
            ui.input_slider("gb_n_trees", "Number of stages:", min=10, max=300, value=100, step=10),
            ui.input_slider("gb_max_depth", "Max tree depth:", min=1, max=6, value=3, step=1),
            ui.input_slider("gb_learning_rate", "Learning rate:", min=0.01, max=0.5, value=0.1, step=0.01),
            ui.input_slider("gb_test_size", "Test set size (%):", min=10, max=40, value=20, step=5),
            col_widths=[3, 3, 3, 3],
        ),
        ui.navset_card_tab(
            ui.nav_panel(
                "Understanding Gradient Boosting",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("What is Gradient Boosting?"),
                        ui.markdown("""
**Gradient Boosting** also builds many trees, but differently
from Random Forest:

1. Start with a simple prediction
2. Build a small tree to fix the **mistakes**
3. Build another tree to fix the **remaining mistakes**
4. Repeat -- each tree focuses on what previous trees got wrong

This **sequential learning** often gives the best accuracy.

Unlike Random Forest (where trees are independent), each
boosting stage **builds on the previous one**.

---

**How does boosting reduce errors?**

Imagine you have a class of students taking a test.
After each test, the teacher looks at which questions
the class got **wrong** and drills them on exactly those
topics. The next test focuses on their **weakest areas**.

Boosting works the same way:
- Each new tree is a new "study session"
- It pays **extra attention** to the data points the
  previous trees got wrong (by giving them higher weight)
- Over many rounds, the combined team of small trees
  becomes extremely accurate -- even though each
  individual tree is very simple (a "weak learner")

*This is why the error drops steeply at first
(the easy mistakes are fixed quickly) and then levels
off as fewer mistakes remain.*
                        """),
                    ),
                    ui.card(
                        ui.card_header("Key hyperparameters"),
                        ui.markdown("""
**Number of stages** -- How many sequential trees to build.
More stages can improve accuracy but risk overfitting.

**Max tree depth** -- How deep each individual tree can grow.
Boosting typically uses **shallow trees** (depth 1--3) because
each tree only needs to fix a small part of the remaining error.

**Learning rate** -- Controls how much each new tree contributes.
A smaller value means each tree has less influence, which usually
needs more stages but produces a more stable model.

*The trade-off: lower learning rate + more stages = slower
training but often better generalisation.*
                        """),
                    ),
                    col_widths=[6, 6],
                ),
            ),
            ui.nav_panel(
                "Feature Importance",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Variable Importance (scaled 0 — 100)"),
                        output_widget("gb_importance"),
                    ),
                    ui.card(
                        ui.card_header("What do these bars mean?"),
                        ui.markdown("""
This chart ranks the **10 global markets** by how useful
each one's *previous-day direction* is for predicting
whether NIFTY will go **UP or DOWN** tomorrow.

**How to read it:**

- The **longest bar (100)** is the single most useful
  market — the model relies on it the most.
- **Shorter bars** mean that market adds less predictive
  value.
- A very short bar means the model barely looks at that
  market when making its prediction.

**How is importance calculated?**

Every time the model adds a new small tree (stage), it
picks the market whose yesterday-direction best separates
upcoming UP days from DOWN days. The chart totals how
much each market contributed to reducing prediction
errors across **all stages combined**.

**Why does this matter?**

It tells you which global markets have the strongest
link to NIFTY's next-day movement. For example, if
"Dow Jones (prev day)" scores 100 and "S&P 500 (prev
day)" scores 19, the Dow's direction yesterday is
roughly **five times more informative** than the S&P's
for forecasting NIFTY tomorrow.

*Try adjusting the sliders above — the rankings can
shift as you change the model's complexity.*
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
                "ROC Curve",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("ROC Curve & AUC"),
                        output_widget("gb_roc_curve"),
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
                "Staged Accuracy",
                ui.markdown("""
Watch how prediction accuracy improves as more trees are added.
Unlike Random Forest (independent trees), each boosting stage
**builds on the previous one**.
                """),
                output_widget("gb_staged"),
            ),
            ui.nav_panel(
                "Boosting vs Error",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("How Boosting Reduces Error Over Iterations"),
                        output_widget("gb_boosting_error"),
                    ),
                    ui.card(
                        ui.card_header("What Does This Chart Show?"),
                        ui.markdown("""
This chart shows how the **test error** (misclassification rate)
decreases as we add more boosting iterations.

**The reference lines** show the error rate of simpler models
trained on the same data:

- **Single Stump** (depth = 1) -- a tree with just one split.
  This is the simplest possible tree, equivalent to asking
  a single yes/no question. It usually has high error.

- **Full Tree** -- a large, fully grown decision tree.
  Despite being complex, it often overfits the training data
  and does not generalise well.

**The boosting curve** (orange) starts near the stump's error
but quickly drops **below both reference lines**. This
demonstrates boosting's key insight:

*Many simple trees working together, each one fixing
the mistakes of the last, can dramatically outperform
a single complex tree.*

The curve flattens out because after many rounds most
easy mistakes have already been corrected and the
remaining errors are harder to fix.
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

def gb_server(input, output, session):

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
        max_imp = importances.max()
        scaled = (importances / max_imp * 100) if max_imp > 0 else importances
        indices = np.argsort(scaled)

        names = [friendly_name(feat_names[i]) for i in indices]
        vals = scaled[indices]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=names, x=vals, orientation='h',
            marker_color='#f59e0b',
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
    def gb_confusion():
        _, _, _, _, y_test, y_pred, _ = gb_model_data()
        return make_confusion_fig(y_test, y_pred, 'Gradient Boosting')

    @output
    @render.ui
    def gb_metrics():
        _, _, _, _, y_test, y_pred, _ = gb_model_data()
        return metrics_html(y_test, y_pred, 'Gradient Boosting')

    @render_widget
    def gb_roc_curve():
        clf, _, X_test, _, y_test, _, _ = gb_model_data()
        y_prob = clf.predict_proba(X_test)[:, 1]
        return make_roc_fig(y_test, y_prob, 'Gradient Boosting')

    @render_widget
    def gb_staged():
        clf, X_train, X_test, y_train, y_test, _, _ = gb_model_data()

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

    @render_widget
    def gb_boosting_error():
        clf, X_train, X_test, y_train, y_test, _, _ = gb_model_data()

        stages = list(range(1, clf.n_estimators + 1))
        test_errors = []
        for y_pred_test in clf.staged_predict(X_test):
            test_errors.append(1 - accuracy_score(y_test, y_pred_test))

        # Reference: single stump (depth=1)
        stump = DecisionTreeClassifier(max_depth=1, random_state=42)
        stump.fit(X_train, y_train)
        stump_error = 1 - accuracy_score(y_test, stump.predict(X_test))

        # Reference: full tree (no max_depth)
        full_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
        full_tree.fit(X_train, y_train)
        full_error = 1 - accuracy_score(y_test, full_tree.predict(X_test))
        full_nodes = full_tree.tree_.node_count

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=stages, y=test_errors, mode='lines',
            name='Boosting', line=dict(color='#ea580c', width=2.5),
        ))

        fig.add_trace(go.Scatter(
            x=[stages[0], stages[-1]], y=[stump_error, stump_error],
            mode='lines', name='Single Stump (depth 1)',
            line=dict(color='#1e293b', width=1.5, dash='dot'),
        ))
        fig.add_annotation(
            x=stages[-1] * 0.65, y=stump_error + 0.015,
            text='Single Stump', showarrow=False,
            font=dict(size=12, color='#1e293b'),
        )

        fig.add_trace(go.Scatter(
            x=[stages[0], stages[-1]], y=[full_error, full_error],
            mode='lines', name=f'{full_nodes} Node Tree',
            line=dict(color='#1e293b', width=1.5, dash='dot'),
        ))
        fig.add_annotation(
            x=stages[-1] * 0.65, y=full_error + 0.015,
            text=f'{full_nodes} Node Tree', showarrow=False,
            font=dict(size=12, color='#1e293b'),
        )

        fig.update_layout(
            height=450,
            xaxis_title='Boosting Iterations',
            yaxis_title='Test Error',
            yaxis_range=[0, max(stump_error, full_error, max(test_errors)) * 1.15],
            legend=dict(orientation='h', y=-0.18),
            margin=dict(l=60, r=30, t=20, b=70),
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
        )
        return fig
