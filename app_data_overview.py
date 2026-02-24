"""
app_data_overview.py — Tab 1: Data Overview
──────────────────────────────────────────
UI panel and server functions for the Data Overview tab.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shiny import render, ui
from shinywidgets import output_widget, render_widget

from app_data import (
    df_model, df_raw_pct, INDEX_NAMES, LAG1_FEATURE_COLS,
    friendly_name,
)


# ═══════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════

def data_overview_ui():
    return ui.nav_panel(
        ui.span(ui.tags.i(class_="bi bi-bar-chart-line me-1"), "Data Overview"),
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
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("When Each Market Was UP vs DOWN Yesterday, What Did NIFTY Do?"),
                            output_widget("eda_conditional_bars"),
                        ),
                        ui.card(
                            ui.card_header("Does yesterday's market direction predict NIFTY today?"),
                            ui.markdown("""
For each global exchange, we look at all the days it went **UP** yesterday
vs all the days it went **DOWN** yesterday, and ask: *what happened to
NIFTY the next day?*

If a market's direction is a useful signal, you'll see a clear difference
between the green and red bars.

- **Green bars** = % of days NIFTY went UP when that exchange was UP yesterday
- **Red bars** = % of days NIFTY went UP when that exchange was DOWN yesterday

A large gap between green and red means that exchange's previous day direction
is a strong signal for NIFTY.
                            """),
                        ),
                        col_widths=[8, 4],
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
    )


# ═══════════════════════════════════════════════════════════
#  Server
# ═══════════════════════════════════════════════════════════

def data_overview_server(input, output, session):

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

No resampling or class weighting needed.
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
                ui.h5("Key Insight"),
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
