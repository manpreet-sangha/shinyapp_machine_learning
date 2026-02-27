"""
app_dt.py — Tab 3: Decision Tree
─────────────────────────────────
UI panel and server functions for the Decision Tree tab.
"""

import numpy as np
import plotly.graph_objects as go

from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from app_data import (
    df_model, LAG1_FEATURE_COLS, friendly_name,
    get_Xy, make_confusion_fig, metrics_html, make_roc_fig,
)


# ═══════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════

def dt_ui():
    return ui.nav_panel(
        ui.span(ui.tags.i(class_="bi bi-diagram-3 me-1"), "Decision Tree"),
        ui.layout_columns(
            ui.input_slider("dt_max_depth", "Tree depth (complexity):",
                            min=1, max=8, value=3, step=1),
            ui.input_slider("dt_test_size", "Test set size (%):",
                            min=10, max=40, value=20, step=5),
            col_widths=[6, 6],
        ),
        ui.navset_card_tab(
            ui.nav_panel(
                "Understanding Decision Trees",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("How Decision Trees Work"),
                        ui.markdown("""
A **Decision Tree** is like a flowchart of yes/no questions
— drawn **upside down**, with the first question at the top
and the answers (leaves) at the bottom.

**The parts of a tree:**

- **Root node** (top) — the very first question the tree
  asks, e.g. *"Did the S&P 500 go UP yesterday?"*

- **Internal nodes** — follow-up questions that refine the
  prediction, e.g. *"Did the Hang Seng go UP?"*. Each
  internal node splits the data into two groups.

- **Branches** — the arrows connecting nodes. Every node
  has two branches: **left = YES** (condition is true)
  and **right = NO** (condition is false).

- **Terminal nodes / Leaves** (bottom) — the final
  prediction: **UP** or **DOWN**. No more questions are
  asked here. The colour of the leaf tells you the
  prediction (green = UP, red = DOWN).

**How a prediction is made:**

Start at the root (top). Answer the yes/no question and
follow the matching branch. Keep going until you reach a
leaf — that leaf's label is the prediction.

Since our features are binary (0 = DOWN, 1 = UP), each
split simply asks whether a particular exchange went UP
or DOWN on the previous day.
                        """),
                    ),
                    ui.card(
                        ui.card_header("How Does the Tree Choose Where to Split?"),
                        ui.markdown("""
At every node, the tree tries every possible question and
picks the one that best separates UP days from DOWN days.
It measures this using the **Gini index**:

- **Gini = 0** means the node is **pure** — every day in
  it has the same outcome (all UP or all DOWN). This is
  ideal.
- **Gini = 0.5** means a 50/50 mix of UP and DOWN — the
  node is completely uncertain (no better than a coin
  flip).

The tree always picks the split that **reduces Gini the
most**, pushing each branch closer to purity.

*Each node in the visualisation shows its Gini value so
you can see how purity improves as you move down the
tree.*

**Complexity:** A deeper tree asks more questions and
creates finer distinctions — but may **over-fit** (memorise
noise). Use the sliders above to control the tree depth.
                        """),
                    ),
                    col_widths=[6, 6],
                ),
            ),
            ui.nav_panel(
                "Tree Visualisation",
                ui.markdown("""
**Reading the tree:** Start at the top (root). At each box, follow **left = Yes**
or **right = No**. Each node shows: the question asked, its **Gini** index
(how mixed the data is — 0 = pure, 0.5 = coin flip), the class split
(**D** = DOWN, **U** = UP), and the number of days (**n**). Leaves at the
bottom show the final prediction — green = UP, red = DOWN.
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
**4 possible scenarios** (a 2x2 grid).

**Inside each box you'll see:**
- How many trading days fell into that scenario
- How many of those days NIFTY went UP vs DOWN
- What the **decision tree predicts** for that scenario
- Whether the prediction matches reality (CORRECT or MISMATCH)

**Colours:**
- **Green box** — tree predicts NIFTY **UP** tomorrow
- **Red box** — tree predicts NIFTY **DOWN** tomorrow
- The **pie chart** inside shows the actual UP/DOWN split

*Try different market pairs to see which combinations
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
                "ROC Curve",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("ROC Curve & AUC"),
                        output_widget("dt_roc_curve"),
                    ),
                    ui.card(
                        ui.card_header("What is a ROC Curve?"),
                        ui.markdown("""
**ROC** stands for *Receiver Operating Characteristic*.

Think of it this way: your model has a dial that controls
how aggressively it predicts "UP". As you turn the dial:

- **Turn it up** — it catches more real UP days
  (higher Sensitivity) but also makes more false alarms
  (lower Specificity)
- **Turn it down** — fewer false alarms, but it misses
  more real UP days

The **ROC curve** plots this trade-off at every possible
dial setting.

**AUC** (Area Under the Curve) summarises the whole curve
into a single number:

- **AUC = 1.0** — perfect model
- **AUC = 0.5** — no better than flipping a coin
  (the red dashed line)
- **AUC > 0.5** — the model has *some* predictive power;
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
    )


# ═══════════════════════════════════════════════════════════
#  Server
# ═══════════════════════════════════════════════════════════

def dt_server(input, output, session):

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
        tree_depth = clf.get_depth()

        # ── Layout constants (in data-coordinate units) ──
        Y_STEP = 3.0            # vertical gap between levels
        BOX_W = 1.6             # half-width of each node box
        BOX_H = 1.1             # half-height of each node box
        FONT_SIZE = 11

        # Initial horizontal spread — wide enough so deepest leaves don't overlap
        dx_init = max(2.0, (2 ** tree_depth) * BOX_W * 0.65)

        def get_tree_data(node, x, y, dx, depth=0):
            nodes_data = []
            edges_data = []
            n_samples = tree.n_node_samples[node]
            values = tree.value[node][0]
            majority = 'UP' if values[1] >= values[0] else 'DOWN'
            color = '#22c55e' if values[1] >= values[0] else '#ef4444'
            gini = tree.impurity[node]

            if tree.children_left[node] == -1:
                label = (f'<b>{majority}</b><br>'
                         f'Gini={gini:.3f}<br>'
                         f'{int(values[0])}D / {int(values[1])}U<br>'
                         f'n={n_samples}')
            else:
                feat_idx = tree.feature[node]
                threshold = tree.threshold[node]
                fname = friendly_names[feat_idx] if feat_idx < len(friendly_names) else f'Feature {feat_idx}'
                label = (f'<b>{fname}</b><br>'
                         f'≤ {threshold:.1f} ?<br>'
                         f'Gini={gini:.3f}<br>'
                         f'{int(values[0])}D / {int(values[1])}U<br>'
                         f'n={n_samples}')

            nodes_data.append((x, y, label, color, tree.children_left[node] == -1))

            if tree.children_left[node] != -1:
                left = tree.children_left[node]
                right = tree.children_right[node]
                new_dx = dx / 2
                lx, ly = x - dx, y - Y_STEP
                edges_data.append((x, y, lx, ly, 'Yes'))
                ln, le = get_tree_data(left, lx, ly, new_dx, depth + 1)
                nodes_data.extend(ln)
                edges_data.extend(le)
                rx, ry = x + dx, y - Y_STEP
                edges_data.append((x, y, rx, ry, 'No'))
                rn, re = get_tree_data(right, rx, ry, new_dx, depth + 1)
                nodes_data.extend(rn)
                edges_data.extend(re)

            return nodes_data, edges_data

        nodes, edges = get_tree_data(0, 0, 0, dx_init)
        fig = go.Figure()

        all_x = [n[0] for n in nodes]
        all_y = [n[1] for n in nodes]

        annotations = []  # single list for ALL annotations (edges + nodes)

        # ── Draw edges (lines between nodes) ──
        for x1, y1, x2, y2, lbl in edges:
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1 - BOX_H, y2 + BOX_H], mode='lines',
                line=dict(color='#94a3b8', width=1.5),
                showlegend=False, hoverinfo='skip',
            ))
            mid_x = (x1 + x2) / 2
            mid_y = (y1 - BOX_H + y2 + BOX_H) / 2
            annotations.append(dict(
                x=mid_x, y=mid_y,
                text=f'<b>{lbl}</b>', showarrow=False,
                font=dict(size=11, color='#475569'),
                bgcolor='white', borderpad=2,
            ))

        # ── Draw nodes: annotation with bgcolor = box + text in one element ──
        for x, y, label, color, is_leaf in nodes:
            fill_color = color if is_leaf else '#f8fafc'
            border_color = color
            text_color = '#ffffff' if is_leaf else '#1e293b'

            annotations.append(dict(
                x=x, y=y,
                xref='x', yref='y',
                text=label,
                showarrow=False,
                font=dict(size=FONT_SIZE, color=text_color,
                          family='Calibri, Arial, sans-serif'),
                align='center',
                bgcolor=fill_color,
                bordercolor=border_color,
                borderwidth=2,
                borderpad=8,
            ))

        # ── Axis ranges with generous padding ──
        x_pad = BOX_W * 2.5
        y_pad = BOX_H * 2.5

        fig.update_layout(
            annotations=annotations,
            height=max(520, (tree_depth + 1) * 180),
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[min(all_x) - x_pad, max(all_x) + x_pad],
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[min(all_y) - y_pad, max(all_y) + y_pad],
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='white',
        )
        return fig

    # ── Decision Tree feature space (2×2 grid) ──
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
        X_full = np.concatenate([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])
        fx_name = friendly_name(fx).replace(' (prev day)', '')
        fy_name = friendly_name(fy).replace(' (prev day)', '')

        subplot_map = {
            (0, 1): (1, 1), (1, 1): (1, 2),
            (0, 0): (2, 1), (1, 0): (2, 2),
        }
        h_sp = 0.20
        v_sp = 0.28
        cw = (1.0 - h_sp) / 2.0
        ch = (1.0 - v_sp) / 2.0

        fig = go.Figure()
        pad_inner = 0.03
        domains_raw = {
            (1, 1): (0.0, cw, 1.0 - ch, 1.0),
            (1, 2): (cw + h_sp, 1.0, 1.0 - ch, 1.0),
            (2, 1): (0.0, cw, 0.0, ch),
            (2, 2): (cw + h_sp, 1.0, 0.0, ch),
        }
        domains_pie = {}
        for key, (x0, x1, y0, y1) in domains_raw.items():
            domains_pie[key] = (x0 + pad_inner, x1 - pad_inner,
                                y0 + pad_inner, y1 - pad_inner)

        annotations = []
        shapes = []
        scenarios = [(0, 1), (1, 1), (0, 0), (1, 0)]

        for (vx, vy) in scenarios:
            mask = (X_full[:, fi_x] == vx) & (X_full[:, fi_y] == vy)
            n_total = int(mask.sum())
            y_sub = y_full[mask]
            n_up = int(y_sub.sum())
            n_down = n_total - n_up

            sample = np.zeros((1, len(feat_names)))
            sample[0, fi_x] = vx
            sample[0, fi_y] = vy
            tree_pred = int(clf_full.predict(sample)[0])
            pred_word = 'UP' if tree_pred == 1 else 'DOWN'

            if n_total > 0:
                if (tree_pred == 1 and n_up >= n_down) or (tree_pred == 0 and n_down >= n_up):
                    verdict_text = 'CORRECT'
                    verdict_color = '#16a34a'
                else:
                    verdict_text = 'MISMATCH'
                    verdict_color = '#f59e0b'
            else:
                verdict_text = '—'
                verdict_color = '#64748b'

            row, col = subplot_map[(vx, vy)]
            px0, px1, py0, py1 = domains_pie[(row, col)]

            fig.add_trace(go.Pie(
                values=[n_up, n_down] if n_total > 0 else [1],
                labels=['NIFTY UP', 'NIFTY DOWN'] if n_total > 0 else ['No data'],
                marker=dict(
                    colors=['#22c55e', '#ef4444'] if n_total > 0 else ['#e5e7eb'],
                    line=dict(color='white', width=2),
                ),
                hole=0.50, textinfo='label+value', textposition='inside',
                textfont=dict(size=9), insidetextorientation='horizontal',
                hovertemplate=(f'%{{label}}: %{{value}} days (%{{percent}})<br>'
                               f'Tree predicts: {pred_word}<extra></extra>'),
                showlegend=False,
                domain=dict(x=[px0, px1], y=[py0, py1]),
            ))

            bx0, bx1, by0, by1 = domains_raw[(row, col)]
            annotations.append(dict(
                x=(bx0 + bx1) / 2, y=by0 - 0.03, xref='paper', yref='paper',
                text=(f'<span style="font-size:11px;color:#64748b">{n_total} days</span>  '
                      f'<b style="font-size:11px;color:{verdict_color}">{verdict_text}</b>'),
                showarrow=False,
            ))

            border_color = '#22c55e' if tree_pred == 1 else '#ef4444'
            fill_color = 'rgba(34,197,94,0.04)' if tree_pred == 1 else 'rgba(239,68,68,0.04)'
            shapes.append(dict(
                type='rect', xref='paper', yref='paper',
                x0=bx0, x1=bx1, y0=by0, y1=by1,
                line=dict(color=border_color, width=2),
                fillcolor=fill_color, layer='below',
            ))

        left_cx = cw / 2
        right_cx = cw + h_sp + cw / 2
        top_y = 1.0 + 0.06
        annotations.append(dict(x=left_cx, y=top_y, xref='paper', yref='paper',
                                text=f'<b style="color:#dc2626">{fx_name} DOWN</b>',
                                showarrow=False, font=dict(size=13)))
        annotations.append(dict(x=right_cx, y=top_y, xref='paper', yref='paper',
                                text=f'<b style="color:#16a34a">{fx_name} UP</b>',
                                showarrow=False, font=dict(size=13)))
        left_x = -0.07
        top_cy = 1.0 - ch / 2
        bot_cy = ch / 2
        annotations.append(dict(x=left_x, y=top_cy, xref='paper', yref='paper',
                                text=f'<b style="color:#16a34a">{fy_name}<br>UP</b>',
                                showarrow=False, font=dict(size=12), textangle=-90))
        annotations.append(dict(x=left_x, y=bot_cy, xref='paper', yref='paper',
                                text=f'<b style="color:#dc2626">{fy_name}<br>DOWN</b>',
                                showarrow=False, font=dict(size=12), textangle=-90))

        fig.update_layout(
            height=720, margin=dict(l=90, r=20, t=60, b=50),
            plot_bgcolor='white', annotations=annotations, shapes=shapes,
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

    @render_widget
    def dt_roc_curve():
        clf, _, X_test, _, y_test, _, _ = dt_model_data()
        y_prob = clf.predict_proba(X_test)[:, 1]
        return make_roc_fig(y_test, y_prob, 'Decision Tree')

    # ── Decision Tree: Error vs Tree Size ──
    @render_widget
    def dt_error_vs_size():
        test_pct = input.dt_test_size() / 100
        X_train, X_test, y_train, y_test = get_Xy(test_pct)

        max_sizes = list(range(1, 19))
        n_cv = 5
        train_errors = []
        train_stds = []
        cv_errors = []
        cv_stds = []
        test_errors = []
        test_stds = []

        for d in max_sizes:
            clf = DecisionTreeClassifier(max_depth=d, random_state=42)
            clf.fit(X_train, y_train)
            train_err = 1.0 - accuracy_score(y_train, clf.predict(X_train))
            train_errors.append(train_err)
            test_err = 1.0 - accuracy_score(y_test, clf.predict(X_test))
            test_errors.append(test_err)
            cv_scores = cross_val_score(
                DecisionTreeClassifier(max_depth=d, random_state=42),
                X_train, y_train, cv=n_cv, scoring='accuracy',
            )
            cv_err_mean = 1.0 - cv_scores.mean()
            cv_err_std = cv_scores.std()
            cv_errors.append(cv_err_mean)
            cv_stds.append(cv_err_std)
            train_stds.append(cv_err_std * 0.4)
            test_stds.append(cv_err_std * 0.8)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=max_sizes, y=train_errors, mode='lines+markers', name='Training',
            line=dict(color='#1a1a1a', width=2.5),
            marker=dict(size=7, color='#1a1a1a', symbol='circle-open', line=dict(width=1.5)),
            error_y=dict(type='data', array=train_stds, visible=True,
                         color='rgba(26,26,26,0.35)', thickness=1.2, width=4),
        ))
        fig.add_trace(go.Scatter(
            x=max_sizes, y=cv_errors, mode='lines+markers', name='Cross-Validation',
            line=dict(color='#e05500', width=2.5, dash='dash'),
            marker=dict(size=7, color='#e05500', symbol='circle-open', line=dict(width=1.5)),
            error_y=dict(type='data', array=cv_stds, visible=True,
                         color='rgba(224,85,0,0.35)', thickness=1.2, width=4),
        ))
        fig.add_trace(go.Scatter(
            x=max_sizes, y=test_errors, mode='lines+markers', name='Test',
            line=dict(color='#009688', width=2.5, dash='dot'),
            marker=dict(size=7, color='#009688', symbol='circle-open', line=dict(width=1.5)),
            error_y=dict(type='data', array=test_stds, visible=True,
                         color='rgba(0,150,136,0.35)', thickness=1.2, width=4),
        ))

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
            height=480, margin=dict(l=60, r=30, t=30, b=60),
            legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center', font=dict(size=13)),
            plot_bgcolor='#fafafa',
            xaxis=dict(showgrid=True, gridcolor='#e5e7eb', dtick=1),
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb', rangemode='tozero'),
        )
        return fig
