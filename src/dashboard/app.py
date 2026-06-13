"""Dash uygulaması — layout + callback kayıtları. Giriş noktası (main).

Çalıştırma: ``python -m src.dashboard.app`` (tarayıcıda http://127.0.0.1:8050).

Uygulama sunucu tarafında bir ``DashboardController`` (dolayısıyla bir
``WarehouseSimulator``) tutar; düğmeler ve interval, controller ilkellerini
çağırır, sonuç 3B figüre ve panellere yansır. Görselleştirme katmanı çekirdeğe
mantık eklemez (docs/06 §6).
"""

from __future__ import annotations

import json
from pathlib import Path

from dash import Dash, Input, Output, State, ctx, dcc, html

from src.dashboard.controllers import ComparisonController, DashboardController
from src.dashboard.panels import (
    build_approaches_panel,
    build_comparison_metrics,
    build_data_exploration,
    build_decision_log,
    build_delay_panel,
    build_heldout_badge,
    build_legend,
    build_log_console,
    build_metric_panel,
    metric_rows,
)
from src.dashboard.warehouse_view import render_warehouse
from src.ml.delay_model import DelayPredictor
from src.simulation.loaders import Scenario

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODEL_PATH: Path = PROJECT_ROOT / "models" / "delay_model.txt"
# PPO için tercih: eval-callback'in seçtiği EN İYİ; yoksa son kaydedilen.
_PPO_BEST_PATH: Path = PROJECT_ROOT / "models" / "ppo_best" / "best_model.zip"
_PPO_LAST_PATH: Path = PROJECT_ROOT / "models" / "ppo_warehouse.zip"
PPO_MODEL_PATH: Path = _PPO_BEST_PATH if _PPO_BEST_PATH.exists() else _PPO_LAST_PATH
# Raf senaryosu modeli (tek kat + affinity'siz eğitilmiş; --rack ile kullanılır).
PPO_RACK_PATH: Path = PROJECT_ROOT / "models" / "ppo_rack.zip"
REPORT_PATH: Path = PROJECT_ROOT / "runs" / "delay_model_report.json"
# Held-out karşılaştırma çıktısı (politika rozetleri için).
COMPARISON_PATH: Path = PROJECT_ROOT / "runs" / "evaluation" / "comparison.json"


def _load_heldout_rehandling() -> dict[str, float]:
    """comparison.json'dan politika→held-out ortalama rehandling sözlüğünü okur (varsa)."""
    if not COMPARISON_PATH.exists():
        return {}
    try:
        report = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
        summary = report.get("summary", {})
        return {p: summary[p]["rehandling"]["mean"] for p in summary}
    except (ValueError, KeyError):
        return {}

# Otonom akışta her interval tıkında kaç adım koşulacağı.
STEPS_PER_TICK: int = 3
# Otomatik oynatma periyodu (ms).
PLAY_INTERVAL_MS: int = 1000


def build_app(
    controller: DashboardController,
    comparison: ComparisonController | None = None,
    delay_mae: float | None = None,
) -> Dash:
    """Verilen controller'a bağlı bir Dash uygulaması kurar (layout + callback'ler).

    ``comparison`` verilirse yan yana politika kıyaslama bölümü (C1) de eklenir.
    Sunucu başlatmaz; test ve yeniden kullanım için uygulama nesnesini döndürür.
    """
    app = Dash(__name__, title="Çelik Depo Dijital İkiz")

    policy_options = [{"label": name, "value": name} for name in controller.policies]
    heldout = _load_heldout_rehandling()  # politika rozetleri için (held-out)

    # ---- Tek-politika canlı görünüm bölümü ----
    single_section = [
        html.H2("Çelik Bobin Depo — Dijital İkiz"),
        # Kontrol çubuğu.
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px",
                   "flexWrap": "wrap", "marginBottom": "10px"},
            children=[
                html.Label("Politika:"),
                dcc.Dropdown(
                    id="policy-dropdown", options=policy_options,
                    value=controller.current_policy_name, clearable=False,
                    style={"width": "180px", "color": "#111"},
                ),
                html.Button("Adım", id="step-btn", n_clicks=0),
                html.Button("Akışı Başlat", id="play-btn", n_clicks=0),
                html.Button("Olay Tetikle (Peak)", id="peak-btn", n_clicks=0),
                html.Button("Sıfırla", id="reset-btn", n_clicks=0),
                # Renklendirme modu (A3): aciliyete göre / bekleme süresine göre.
                html.Label("Renk:", style={"marginLeft": "12px"}),
                dcc.RadioItems(
                    id="color-mode",
                    options=[{"label": " Aciliyet", "value": "urgency"},
                             {"label": " Bekleme", "value": "dwell"}],
                    value="urgency", inline=True,
                    style={"display": "inline-block"},
                ),
                # Ufuk seçici: sim ne kadar uzun koşsun (doluluk buna bağlı).
                html.Label("Ufuk:", style={"marginLeft": "12px"}),
                dcc.Dropdown(
                    id="horizon-select",
                    options=[{"label": "24 saat (~%50)", "value": 24},
                             {"label": "48 saat (~%65)", "value": 48},
                             {"label": "72 saat (~%75)", "value": 72}],
                    value=48, clearable=False,
                    style={"width": "160px", "color": "#111"},
                ),
                # Senaryo seçici: Ana (2 kat + affinity) ↔ Raf (tek kat, rota). Raf'ta PPO
                # raf modeline geçer; demoda komut satırı yerine canlı geçiş.
                html.Label("Senaryo:", style={"marginLeft": "12px"}),
                dcc.Dropdown(
                    id="scenario-select",
                    options=[{"label": "Ana (2 kat + affinity)", "value": "main"},
                             {"label": "Raf (tek kat, rota)", "value": "rack"}],
                    value="rack" if controller.rack_mode else "main", clearable=False,
                    style={"width": "190px", "color": "#111"},
                ),
            ],
        ),
        # Seçili politikanın held-out genel performansı (#1 rozet).
        html.Div(
            id="heldout-badge", style={"marginBottom": "10px"},
            children=build_heldout_badge(controller.current_policy_name, heldout),
        ),
        # 3B depo görünümü — BÜYÜK, tam genişlik (ekranı kaplar).
        dcc.Graph(
            id="warehouse-graph",
            style={"height": "70vh", "minHeight": "540px", "width": "100%"},
            config={"displaylogo": False, "scrollZoom": True},
        ),
        # Gösterge açıklaması (A1) + yaklaşımlar (C3): simin ALTINDA, YATAY yan yana.
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "margin": "10px 0"},
            children=[
                html.Div(build_legend(), style={"flex": "1 1 300px"}),
                html.Div(build_approaches_panel(), style={"flex": "2 1 420px"}),
            ],
        ),
        # Metrik paneli (D2 doyum uyarısı içerir).
        html.H4("Metrikler"),
        html.Div(id="metric-panel"),
        # Karar günlüğü (B2), olay günlüğü ve canlı gecikme (ML tahmini vs gerçek) yan yana.
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            children=[
                html.Div([html.H4("Karar Günlüğü"), html.Div(id="decision-log")],
                         style={"flex": "1", "minWidth": "300px"}),
                html.Div([html.H4("Olay Günlüğü"), html.Div(id="log-console")],
                         style={"flex": "1", "minWidth": "300px"}),
                html.Div([html.H4("Canlı Gecikme (ML tahmini ↔ gerçek)"),
                          html.Div(id="delay-panel")],
                         style={"flex": "1", "minWidth": "320px"}),
            ],
        ),
        # Otonom akış zamanlayıcısı (başlangıçta kapalı).
        dcc.Interval(id="interval", interval=PLAY_INTERVAL_MS, n_intervals=0, disabled=True),
        # Veri keşfi (statik): üretilen verinin gecikme örüntüleri — ML neyi öğreniyor.
        html.Hr(style={"margin": "20px 0", "borderColor": "#333"}),
        html.H2("Veri Keşfi — Üretilen Verinin Örüntüleri"),
        build_data_exploration(controller.scenario),
    ]

    children = list(single_section)
    if comparison is not None:
        children += _comparison_section(comparison)

    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "background": "#15151c", "color": "#e0e0e8",
               "padding": "12px", "minHeight": "100vh"},
        children=children,
    )

    @app.callback(
        Output("warehouse-graph", "figure"),
        Output("metric-panel", "children"),
        Output("decision-log", "children"),
        Output("log-console", "children"),
        Output("heldout-badge", "children"),
        Output("delay-panel", "children"),
        Input("step-btn", "n_clicks"),
        Input("interval", "n_intervals"),
        Input("peak-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        Input("policy-dropdown", "value"),
        Input("color-mode", "value"),
        Input("horizon-select", "value"),
        Input("scenario-select", "value"),
    )
    def _update(step_clicks, n_intervals, peak_clicks, reset_clicks, policy_value, color_mode, horizon, scenario):
        """Tüm kontrolleri tek noktadan işler; tetikleyene göre controller'ı çağırır."""
        trigger = ctx.triggered_id
        if trigger == "policy-dropdown" and policy_value:
            controller.set_policy(policy_value)
        elif trigger == "horizon-select" and horizon:
            controller.set_horizon(horizon)  # ufuk değişince sim sıfırlanır
        elif trigger == "scenario-select" and scenario:
            controller.set_rack_mode(scenario == "rack")  # senaryo geçişi: model swap + reset
        elif trigger == "step-btn":
            controller.step()
        elif trigger == "interval":
            controller.run_steps(STEPS_PER_TICK)
        elif trigger == "peak-btn":
            controller.trigger_peak()
        elif trigger == "reset-btn":
            controller.reset()
        # color-mode tetiklerse yalnızca yeniden çizim olur (durum ilerlemez).
        # İlk yüklemede (trigger None) yalnızca mevcut durum çizilir.

        controller.stamp_all_urgency()
        fill = controller.sim.state.fill_ratio()
        # Sim saatini + renk modunu geçir → dwell hover (A3) ve renklendirme modu (A3).
        figure = render_warehouse(controller.sim.state, controller.sim.layout,
                                  now=controller.sim.clock, color_mode=color_mode or "urgency")
        rows = metric_rows(
            controller.sim.metrics, controller.current_policy_name,
            controller.sim.clock, fill, delay_mae,
        )
        return (
            figure,
            build_metric_panel(rows, fill_ratio=fill),  # D2: doyum uyarısı için fill geçir
            build_decision_log(controller.decisions),    # B2
            build_log_console(controller.log),
            build_heldout_badge(controller.current_policy_name, heldout),  # #1 rozet
            build_delay_panel(controller.active_delays()),  # canlı gecikme (ML↔gerçek)
        )

    @app.callback(
        Output("interval", "disabled"),
        Output("play-btn", "children"),
        Input("play-btn", "n_clicks"),
        State("interval", "disabled"),
        prevent_initial_call=True,
    )
    def _toggle_play(n_clicks, currently_disabled):
        """Otonom akışı başlatır/durdurur (interval'i aç/kapat)."""
        now_disabled = not currently_disabled
        return now_disabled, ("Akışı Başlat" if now_disabled else "Akışı Durdur")

    if comparison is not None:
        _register_comparison_callbacks(app, comparison)

    return app


def _comparison_section(comparison: ComparisonController) -> list:
    """Yan yana politika kıyaslama bölümünün layout'unu üretir (C1)."""
    opts = [{"label": n, "value": n} for n in comparison.policies]
    return [
        html.Hr(style={"margin": "20px 0", "borderColor": "#333"}),
        html.H2("Politika Kıyaslama — Yan Yana"),
        html.Div(
            "Aynı tohumlanmış senaryo iki politikayla EŞZAMANLI koşar (paired); "
            "fark yalnızca takılı politikadır. Politika değişimi her iki şeridi t=0'a alır.",
            style={"color": "#9aa0b0", "fontSize": "12px", "marginBottom": "8px"},
        ),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px",
                   "flexWrap": "wrap", "marginBottom": "10px"},
            children=[
                html.Label("A:"),
                dcc.Dropdown(id="cmp-policy-a", options=opts, value=comparison.policy_a,
                             clearable=False, style={"width": "160px", "color": "#111"}),
                html.Label("B:"),
                dcc.Dropdown(id="cmp-policy-b", options=opts, value=comparison.policy_b,
                             clearable=False, style={"width": "160px", "color": "#111"}),
                html.Button("Adım", id="cmp-step-btn", n_clicks=0),
                html.Button("Akışı Başlat", id="cmp-play-btn", n_clicks=0),
                html.Button("Olay Tetikle (Peak)", id="cmp-peak-btn", n_clicks=0),
                html.Button("Sıfırla", id="cmp-reset-btn", n_clicks=0),
                html.Label("Ufuk:", style={"marginLeft": "8px"}),
                dcc.Dropdown(id="cmp-horizon", options=[{"label": "24s", "value": 24},
                             {"label": "48s", "value": 48}, {"label": "72s", "value": 72}],
                             value=48, clearable=False, style={"width": "90px", "color": "#111"}),
                html.Label("Senaryo:", style={"marginLeft": "8px"}),
                dcc.Dropdown(id="cmp-scenario-select",
                             options=[{"label": "Ana (2 kat)", "value": "main"},
                                      {"label": "Raf (tek kat)", "value": "rack"}],
                             value="rack" if comparison.rack_mode else "main", clearable=False,
                             style={"width": "150px", "color": "#111"}),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "10px"},
            children=[
                dcc.Graph(id="cmp-graph-a", style={"height": "420px", "flex": "1"}),
                dcc.Graph(id="cmp-graph-b", style={"height": "420px", "flex": "1"}),
            ],
        ),
        html.H4("Canlı Metrik Kıyaslaması"),
        html.Div(id="cmp-metrics"),
        dcc.Interval(id="cmp-interval", interval=PLAY_INTERVAL_MS, n_intervals=0, disabled=True),
    ]


def _register_comparison_callbacks(app: Dash, comparison: ComparisonController) -> None:
    """Kıyaslama bölümünün callback'lerini kaydeder (C1)."""

    @app.callback(
        Output("cmp-graph-a", "figure"),
        Output("cmp-graph-b", "figure"),
        Output("cmp-metrics", "children"),
        Input("cmp-step-btn", "n_clicks"),
        Input("cmp-interval", "n_intervals"),
        Input("cmp-peak-btn", "n_clicks"),
        Input("cmp-reset-btn", "n_clicks"),
        Input("cmp-policy-a", "value"),
        Input("cmp-policy-b", "value"),
        Input("cmp-horizon", "value"),
        Input("cmp-scenario-select", "value"),
    )
    def _cmp_update(step, n_int, peak, reset, pol_a, pol_b, horizon, scenario):
        trigger = ctx.triggered_id
        if trigger == "cmp-policy-a" and pol_a:
            comparison.set_policy_a(pol_a)
        elif trigger == "cmp-policy-b" and pol_b:
            comparison.set_policy_b(pol_b)
        elif trigger == "cmp-horizon" and horizon:
            comparison.set_horizon(horizon)  # her iki şerit ufku + sıfırla
        elif trigger == "cmp-scenario-select" and scenario:
            comparison.set_rack_mode(scenario == "rack")  # her iki şerit senaryo geçişi
        elif trigger == "cmp-step-btn":
            comparison.step()
        elif trigger == "cmp-interval":
            comparison.run_steps(STEPS_PER_TICK)
        elif trigger == "cmp-peak-btn":
            comparison.trigger_peak()
        elif trigger == "cmp-reset-btn":
            comparison.reset()

        comparison.lane_a.stamp_all_urgency()
        comparison.lane_b.stamp_all_urgency()
        fig_a = render_warehouse(comparison.lane_a.sim.state, comparison.lane_a.sim.layout,
                                 now=comparison.lane_a.sim.clock)
        fig_b = render_warehouse(comparison.lane_b.sim.state, comparison.lane_b.sim.layout,
                                 now=comparison.lane_b.sim.clock)
        metrics = build_comparison_metrics(
            comparison.lane_metrics(comparison.lane_a),
            comparison.lane_metrics(comparison.lane_b),
        )
        return fig_a, fig_b, metrics

    @app.callback(
        Output("cmp-interval", "disabled"),
        Output("cmp-play-btn", "children"),
        Input("cmp-play-btn", "n_clicks"),
        State("cmp-interval", "disabled"),
        prevent_initial_call=True,
    )
    def _cmp_toggle_play(n_clicks, currently_disabled):
        now_disabled = not currently_disabled
        return now_disabled, ("Akışı Başlat" if now_disabled else "Akışı Durdur")


def _load_controller(rack_mode: bool = False) -> tuple[DashboardController, ComparisonController, float | None]:
    """data/ ve eğitilmiş modelden tek-politika + kıyaslama controller'larını kurar.

    ``rack_mode`` True ise: raf senaryosu (tek kat + affinity'siz) ve raf modeli
    (``ppo_rack.zip``) yüklenir; PPO'nun kazandığı saf rota-optimizasyonu demosu.
    """
    scenario = Scenario.from_data_dir()
    delay_model = DelayPredictor.load(str(MODEL_PATH)) if MODEL_PATH.exists() else None
    delay_mae: float | None = None
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        delay_mae = report.get("test_metrics", {}).get("mae")
    # ANA (2 kat) ve RAF (tek kat) PPO modellerini BİR KEZ yükle; arayüzdeki "Senaryo"
    # seçici canlı geçişte ikisi arasında swap eder. Her biri controller'lara paylaştırılır
    # (aksi hâlde ağır modeller defalarca yüklenir).
    main_path = str(PPO_MODEL_PATH) if PPO_MODEL_PATH.exists() else None
    rack_path = str(PPO_RACK_PATH) if PPO_RACK_PATH.exists() else None
    ppo_main = ppo_rack = None
    if main_path or rack_path:
        from sb3_contrib import MaskablePPO

        if main_path:
            ppo_main = MaskablePPO.load(main_path)
        if rack_path:
            ppo_rack = MaskablePPO.load(rack_path)
    if rack_mode and rack_path is None:
        print(f"UYARI: raf modeli bulunamadı ({PPO_RACK_PATH}); raf modunda ana model kullanılır.")
    shared = dict(delay_model=delay_model, ppo_model_path=main_path, ppo_model=ppo_main,
                  ppo_rack_path=rack_path, ppo_model_rack=ppo_rack, rack_mode=rack_mode)
    controller = DashboardController(scenario, **shared)
    comparison = ComparisonController(scenario, **shared)
    return controller, comparison, delay_mae


def main() -> None:
    """Dash sunucusunu başlatır (http://127.0.0.1:8050).

    ``--rack``: raf senaryosu modu (tek kat + affinity'siz, raf modeli) — PPO'nun
    kazandığı saf rota-optimizasyonu demosu. Bayraksız: ana senaryo (2 kat + affinity).
    """
    import argparse
    parser = argparse.ArgumentParser(description="Çelik depo dashboard")
    parser.add_argument("--rack", action="store_true",
                        help="Raf senaryosu modu (tek kat + affinity'siz, raf modeli yüklenir)")
    args = parser.parse_args()
    mod = "RAF SENARYOSU (tek kat, affinity yok, raf modeli)" if args.rack else "ANA SENARYO (2 kat + affinity)"
    print(f">>> Dashboard modu: {mod}")
    controller, comparison, delay_mae = _load_controller(rack_mode=args.rack)
    app = build_app(controller, comparison=comparison, delay_mae=delay_mae)
    app.run(debug=False, port=8050)


if __name__ == "__main__":
    main()
