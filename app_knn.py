"""
app_knn.py — Tab 2: Best Predictor (kNN)
────────────────────────────────────────
UI panel and server functions for the kNN tab.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shiny import render, ui
from shinywidgets import output_widget, render_widget
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from app_data import df_model, LAG1_FEATURE_COLS, friendly_name


# ═══════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════

def knn_ui():
    return ui.nav_panel(
        ui.span(ui.tags.i(class_="bi bi-bullseye me-1"), "Best Predictor (kNN)"),
        ui.navset_card_tab(
            # ── Sub-tab 1: Individual predictor comparison ──
            ui.nav_panel(
                "Predictor Comparison",
                ui.layout_columns(
                    # ── Left: explanation + chart stacked ──
                    ui.div(
                        ui.card(
                            ui.card_header("How kNN Works"),
                            ui.markdown("""
**Goal:** Find which global exchange best predicts NIFTY's next-day direction.

**Method — k-Nearest Neighbours:** For each new day, kNN looks at the **k most
similar past days** and checks what NIFTY did on those days. If most went UP,
it predicts UP; otherwise DOWN. Each bar below tests one exchange as the only
clue — taller bars mean that exchange is a better predictor.
                            """),
                            ui.input_slider("knn_k", "Number of neighbours (k):",
                                            min=1, max=15, value=5, step=2),
                        ),
                        ui.card(
                            ui.card_header("kNN Accuracy by Individual Predictor"),
                            output_widget("knn_predictor_chart"),
                        ),
                    ),
                    # ── Right: key findings ──
                    ui.card(
                        ui.card_header("Key Findings"),
                        ui.output_ui("knn_findings"),
                    ),
                    col_widths=[8, 4],
                ),
            ),
            # ── Sub-tab 2: Which k is good? ──
            ui.nav_panel(
                "Which k is Good?",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Train vs Test Error for Different k"),
                        output_widget("knn_k_vs_error_chart"),
                    ),
                    ui.card(
                        ui.card_header("How to Read This Chart"),
                        ui.output_ui("knn_k_explanation"),
                    ),
                    col_widths=[7, 5],
                ),
            ),
            # ── Sub-tab 3: Accuracy Reliability (Boxplot) ──
            ui.nav_panel(
                "Accuracy Reliability",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Accuracy Distribution — Repeated Random Splits"),
                        output_widget("knn_accuracy_boxplot"),
                    ),
                    ui.card(
                        ui.card_header("What Does This Boxplot Tell Us?"),
                        ui.output_ui("knn_boxplot_summary"),
                    ),
                    col_widths=[3, 9],
                ),
            ),
            # ── Sub-tab 4: Understanding kNN ──
            ui.nav_panel(
                "Understanding kNN",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Which k Should You Choose?"),
                        ui.markdown("""
The value of **k** (how many neighbours to consult) is the **tuning parameter**
of kNN — picking the right k is crucial.

- **Small k (e.g. 1-3):** Reacts to every tiny pattern, including random noise.
  This is **over-fitting** — the model memorises the training data but guesses
  poorly on new, unseen data.

- **Large k (e.g. 15+):** Averages out all useful detail and becomes too simple.
  This is **under-fitting** — the model misses real patterns.

- **Just right (middle range):** The sweet spot is where the validation error
  is **lowest**.
                        """),
                    ),
                    ui.card(
                        ui.card_header("How to Tune k Properly"),
                        ui.markdown("""
We need a fair way to test different values of k — but there are two things
we **cannot** do:

- **Cannot use the test set** — that data is kept locked away until the very
  end. If we peeked at it to pick k, the final score would be unfairly
  optimistic (this is called *data leakage*).
- **Cannot use the training error** — the model will always look good on data
  it has already seen, so the training error is misleadingly low.

The solution is **cross-validation** — a way to create a mini "practice test"
from the training data itself:

1. Split the training data into **5 equal slices** (folds).
2. Take one slice out and pretend it is the "practice test". Train the model on
   the other 4 slices and check how many predictions it gets right on the
   held-out slice.
3. Rotate — repeat this 5 times so every slice gets a turn as the practice test.
4. Average the 5 scores to get one reliable accuracy number.

We do this for every candidate k (1, 3, 5, ..., 25) and keep the k that scores
**highest on average**.
                        """),
                    ),
                    col_widths=[6, 6],
                ),
                ui.card(
                    ui.card_header("Why 5 Folds?"),
                    ui.markdown("""
5 is the most widely used default in machine learning. It gives a good balance:
enough training data in each round (80%) to learn patterns, and enough
validation data (20%) for a meaningful check. Using fewer folds (e.g. 2-3)
leaves less data for training; using many more (e.g. 10+) is slower and can
produce noisier estimates.
                    """),
                ),
            ),
        ),
    )


# ═══════════════════════════════════════════════════════════
#  Server
# ═══════════════════════════════════════════════════════════

def knn_server(input, output, session):

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
            height=len(res_df) * 36 + 60,
            xaxis_title='Accuracy (5-fold CV)',
            xaxis_range=[0.35, 0.70],
            margin=dict(l=160, r=80, t=10, b=40),
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
                ui.h5("Best Single Predictor"),
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

    # ── kNN: k vs Error Rate chart ──
    @output
    @render_widget
    def knn_k_vs_error_chart():
        y = df_model['NIFTY_Direction'].values
        X = df_model[LAG1_FEATURE_COLS].values

        k_range = list(range(1, 26, 2))  # 1,3,5,...,25
        train_errors = []
        test_errors = []

        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            # Test error via cross-validation
            cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
            test_errors.append(1 - cv_scores.mean())
            # Train error: fit on full data, predict on full data
            knn.fit(X, y)
            train_acc = knn.score(X, y)
            train_errors.append(1 - train_acc)

        # Find best k (lowest test error)
        best_idx = test_errors.index(min(test_errors))
        best_k = k_range[best_idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=k_range, y=train_errors,
            mode='lines+markers', name='Training Error',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=k_range, y=test_errors,
            mode='lines+markers', name='Test Error (CV)',
            line=dict(color='#ef4444', width=2),
            marker=dict(size=7),
        ))
        # Mark the best k
        fig.add_vline(x=best_k, line_dash='dot', line_color='#22c55e',
                      annotation_text=f'Best k = {best_k}',
                      annotation_position='top right',
                      annotation_font_color='#166534')

        fig.update_layout(
            xaxis_title='Number of Neighbours (k)',
            yaxis_title='Error Rate (1 − Accuracy)',
            xaxis=dict(dtick=2),
            height=420,
            margin=dict(l=60, r=40, t=30, b=50),
            legend=dict(x=0.65, y=0.98),
        )
        return fig

    @output
    @render.ui
    def knn_k_explanation():
        y = df_model['NIFTY_Direction'].values
        X = df_model[LAG1_FEATURE_COLS].values

        k_range = list(range(1, 26, 2))
        test_errors = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
            test_errors.append(1 - cv_scores.mean())
        best_idx = test_errors.index(min(test_errors))
        best_k = k_range[best_idx]
        best_err = test_errors[best_idx]

        return ui.div(
            ui.div(
                ui.h5("Optimal k"),
                ui.div(f"k = {best_k}", style="font-size:2em; font-weight:bold; color:#166534;"),
                ui.p(f"Test error: {best_err:.1%}  |  Accuracy: {1-best_err:.1%}"),
                style="background:#f0fdf4; border-radius:8px; padding:16px; text-align:center; margin-bottom:12px;",
            ),
            ui.markdown(f"""
**How did we find the best k?**

We tried every odd value of k from 1 to 25. For each one
we ran **5-fold cross-validation**: the training data was
split into 5 slices, and the model was tested on each
slice in turn while learning from the other four. The
5 scores were averaged to give a fair accuracy estimate.

The k that produced the **lowest average error** wins —
that turned out to be **k = {best_k}**.

**Why 5 folds?** It is the standard default in machine
learning — each round uses 80% of the data for training
and 20% for validation, which is a good balance between
having enough data to learn and enough to test.

**What happens at the extremes?**

- **k = 1 (far left):** The model copies the training
  data almost perfectly, so the blue line is near zero.
  But it falls apart on new data — the red line is high.
  This gap means **over-fitting**.

- **k = 25 (far right):** The model is too blunt — it
  asks so many neighbours that the answer is always
  roughly the same. Both lines are high because it
  **under-fits**.

- **k = {best_k} (green line):** The red line dips to
  its lowest point — the model has found the right
  balance.

**Quick guide to the chart:**

Blue line = training error (how wrong on data it learned
from). Red line = validation error (how wrong on unseen
data). When the gap is **small**, the model generalises
well. The sweet spot is where the red line is lowest.
            """),
        )

    # ── kNN: Accuracy Boxplot (repeated random splits) ──
    @render_widget
    def knn_accuracy_boxplot():
        k = input.knn_k()
        y = df_model['NIFTY_Direction'].values
        X = df_model[LAG1_FEATURE_COLS].values

        # 30 repeated random 80/20 splits for a robust distribution
        n_repeats = 30
        accuracies = []
        for i in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=i, stratify=y
            )
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            accuracies.append(knn.score(X_test, y_test))

        acc_arr = np.array(accuracies)

        fig = go.Figure()
        fig.add_trace(go.Box(
            y=acc_arr,
            name='',
            boxmean=True,
            marker_color='#ef4444',
            fillcolor='#ef4444',
            line=dict(color='#1e1e1e'),
            boxpoints='outliers',
            jitter=0.3,
            width=0.5,
        ))
        fig.update_layout(
            yaxis_title='Classification Accuracy',
            height=520,
            margin=dict(l=50, r=15, t=15, b=30),
            yaxis=dict(tickformat='.2f'),
            xaxis=dict(showticklabels=False),
            showlegend=False,
        )
        return fig

    @output
    @render.ui
    def knn_boxplot_summary():
        k = input.knn_k()
        y = df_model['NIFTY_Direction'].values
        X = df_model[LAG1_FEATURE_COLS].values

        n_repeats = 30
        accuracies = []
        for i in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=i, stratify=y
            )
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            accuracies.append(knn.score(X_test, y_test))

        acc_arr = np.array(accuracies)
        q1 = np.percentile(acc_arr, 25)
        median = np.median(acc_arr)
        q3 = np.percentile(acc_arr, 75)
        mean = acc_arr.mean()
        iqr = q3 - q1
        min_val = acc_arr.min()
        max_val = acc_arr.max()

        return ui.div(
            ui.markdown(f"""
**Why a boxplot?**

A single accuracy number can be misleading — it depends on
which days happen to land in the test set. By repeating the
experiment **{n_repeats} times** with different random
80/20 splits, we see the **full range** of accuracies the
model can produce. The boxplot summarises this distribution.
            """),
            ui.hr(),
            ui.div(
                ui.h5("Summary Statistics"),
                ui.tags.table(
                    ui.tags.tr(ui.tags.td("Median", style="padding:4px 12px;"),
                               ui.tags.td(f"{median:.1%}", style="padding:4px 12px; font-weight:bold;")),
                    ui.tags.tr(ui.tags.td("Mean", style="padding:4px 12px;"),
                               ui.tags.td(f"{mean:.1%}", style="padding:4px 12px; font-weight:bold;")),
                    ui.tags.tr(ui.tags.td("Lower Quartile (Q1)", style="padding:4px 12px;"),
                               ui.tags.td(f"{q1:.1%}", style="padding:4px 12px; font-weight:bold;")),
                    ui.tags.tr(ui.tags.td("Upper Quartile (Q3)", style="padding:4px 12px;"),
                               ui.tags.td(f"{q3:.1%}", style="padding:4px 12px; font-weight:bold;")),
                    ui.tags.tr(ui.tags.td("IQR (Q3 − Q1)", style="padding:4px 12px;"),
                               ui.tags.td(f"{iqr:.1%}", style="padding:4px 12px; font-weight:bold;")),
                    ui.tags.tr(ui.tags.td("Min", style="padding:4px 12px;"),
                               ui.tags.td(f"{min_val:.1%}", style="padding:4px 12px; font-weight:bold;")),
                    ui.tags.tr(ui.tags.td("Max", style="padding:4px 12px;"),
                               ui.tags.td(f"{max_val:.1%}", style="padding:4px 12px; font-weight:bold;")),
                    style="width:100%; border-collapse:collapse;",
                ),
                style="background:#f8fafc; border-radius:8px; padding:12px; margin-bottom:12px;",
            ),
            ui.markdown(f"""
**How to read these numbers:**

- **Median ({median:.1%}):** The "middle" accuracy — half
  the runs scored above this, half below. This is often
  more reliable than the mean because it is not pulled
  by outliers.

- **Mean ({mean:.1%}):** The average across all {n_repeats}
  runs. {'Close to the median, so the distribution is fairly symmetric.' if abs(mean - median) < 0.005 else 'Differs from the median, suggesting some skewed or outlier runs.'}

- **Lower Quartile / Q1 ({q1:.1%}):** 25% of runs scored
  below this. Think of it as the "bad-luck" scenario.

- **Upper Quartile / Q3 ({q3:.1%}):** 25% of runs scored
  above this. This is the "good-luck" scenario.

- **IQR ({iqr:.1%}):** The spread of the middle 50% of
  runs. {'A narrow IQR means the model is **stable** across different splits.' if iqr < 0.05 else 'A wider IQR suggests the model is **sensitive** to which days end up in the test set.'}

Any dots above or below the whiskers are **outliers** —
unusually good or bad runs caused by a particular split.
            """),
        )
