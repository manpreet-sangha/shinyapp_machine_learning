"""
app.py — Shiny Core App (Orchestrator)
──────────────────────────────────────
Slim entry point that imports UI panels and server logic
from modular component files, then assembles the full app.

Run with:  shiny run app.py
"""

from shiny import App, Inputs, Outputs, Session, ui

# ── Component modules ──
from app_data_overview import data_overview_ui, data_overview_server
from app_knn import knn_ui, knn_server
from app_dt import dt_ui, dt_server
from app_rf import rf_ui, rf_server
from app_gb import gb_ui, gb_server
from app_cm import cm_ui, cm_server


# ═══════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════
app_ui = ui.page_navbar(
    # ── Responsive CSS for all devices ──
    ui.head_content(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
        ),
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

    # ── Tab panels from component modules ──
    data_overview_ui(),   # Tab 1
    knn_ui(),             # Tab 2
    dt_ui(),              # Tab 3
    rf_ui(),              # Tab 4
    gb_ui(),              # Tab 5
    cm_ui(),              # Tab 6

    title="NIFTY 50 Direction Predictor — Tree-Based ML Methods",
    id="main_nav",
    navbar_options=ui.navbar_options(bg="#1a1a2e", theme="dark"),
)


# ═══════════════════════════════════════════════════════════
#  Server
# ═══════════════════════════════════════════════════════════
def server(input: Inputs, output: Outputs, session: Session):
    data_overview_server(input, output, session)
    knn_server(input, output, session)
    dt_server(input, output, session)
    rf_server(input, output, session)
    gb_server(input, output, session)
    cm_server(input, output, session)


# ═══════════════════════════════════════════════════════════
#  Create app
# ═══════════════════════════════════════════════════════════
app = App(app_ui, server)
