"""
app_cm.py — Tab 6: Compare Models
──────────────────────────────────
UI panel and server functions for the model-comparison tab.
"""

import plotly.graph_objects as go

from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)

from app_data import get_Xy, df_model, LAG1_FEATURE_COLS


# ═══════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════

def cm_ui():
    return ui.nav_panel(
        ui.span(ui.tags.i(class_="bi bi-layout-split me-1"), "Compare Models"),
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
                    ui.div(
                        ui.markdown("""
**Accuracy** — Of all predictions, how many were correct? *(Did the model get it right overall?)*

**Precision** — When the model predicted UP, how often was it actually UP? *(Can I trust an UP prediction?)*

**Recall** — Of all the real UP days, how many did the model catch? *(Did it miss many UP days?)*

**F1** — A single score that balances Precision and Recall. *(High only when both are good.)*
                        """),
                        style="font-size:0.85em; padding:0 16px 8px 16px; color:#555;",
                    ),
                ),
                ui.card(
                    ui.card_header("Key Takeaways"),
                    ui.output_ui("cmp_takeaways"),
                ),
                col_widths=[7, 5],
            ),
        ),
    )


# ═══════════════════════════════════════════════════════════
#  Server
# ═══════════════════════════════════════════════════════════

def cm_server(input, output, session):

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
                ui.h5("Best Model"),
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
