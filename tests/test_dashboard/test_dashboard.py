"""Aşama 5 dashboard'unun kabul kriterleri testleri (docs/06 §8).

Arayüz testleri hafiftir; ağırlık figür üreticisi ve controller köprüsündedir.
Senaryo bellekte elle kurulur (data/ klasörüne ve tarayıcıya bağımlı değildir).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import plotly.graph_objects as go

from src.domain import (
    CoilStatus, CoilType, LogisticsLine, Order, OrderPriority, OrderStatus,
    QualityClass, SlotCoord, SteelCoil, Vehicle, VehicleType, Weather, WarehouseLayout,
)
from src.dashboard.controllers import DashboardController
from src.dashboard.panels import build_legend, metric_rows
from src.dashboard.warehouse_view import (
    FILLED_TRACE_NAME, REHANDLED_TRACE_NAME, render_warehouse,
)
from src.simulation.loaders import Scenario
from src.simulation.metrics import SimulationMetrics
from src.simulation.warehouse_state import WarehouseState

_PROD = datetime(2025, 6, 1, 12, 0)


def _layout() -> WarehouseLayout:
    return WarehouseLayout(
        2, 2, 3,
        {0: LogisticsLine.SHIP_1, 1: LogisticsLine.TRAIN_A},
        {0: 800.0, 1: 800.0},
        (0, 0),
    )


def _coil(coil_id: str, weight: float, urgency: float = 0.0, order_id: str | None = "O1") -> SteelCoil:
    coil = SteelCoil(
        coil_id, CoilType.COLD_ROLLED, weight, 1000, 1400, QualityClass.B, 3,
        _PROD, order_id, CoilStatus.PENDING_PLACEMENT, None, 0.0,
    )
    coil.urgency_score = urgency
    return coil


def test_render_warehouse_valid():
    """render_warehouse geçerli bir go.Figure döndürür; dolu nokta sayısı tutarlı."""
    layout = _layout()
    state = WarehouseState(layout)
    state.place(_coil("C1", 20.0, 0.5), SlotCoord(0, 0, 0))
    state.place(_coil("C2", 12.0, 0.5), SlotCoord(1, 1, 0))

    figure = render_warehouse(state, layout)
    assert isinstance(figure, go.Figure)
    filled = [t for t in figure.data if t.name == FILLED_TRACE_NAME][0]
    assert len(filled.x) == state.occupied_count() == 2


def test_dwell_time_in_hover_when_now_given():
    """now verilince hover'da bobinin bekleme süresi (dwell) görünür; verilmezse görünmez."""
    layout = _layout()
    state = WarehouseState(layout)
    coil = _coil("C1", 20.0, 0.5)
    coil.stored_at = 2.0  # t=2'de depoya girdi
    state.place(coil, SlotCoord(0, 0, 0))

    # now=7.5 -> bekleme 5.5 saat hover'da olmalı.
    fig = render_warehouse(state, layout, now=7.5)
    filled = [t for t in fig.data if t.name == FILLED_TRACE_NAME][0]
    assert "Bekleme: 5.5 sa" in filled.text[0]
    # now verilmezse bekleme satırı eklenmez (geriye uyumluluk).
    fig2 = render_warehouse(state, layout)
    filled2 = [t for t in fig2.data if t.name == FILLED_TRACE_NAME][0]
    assert "Bekleme" not in filled2.text[0]


def test_color_mode_dwell_differs_from_urgency():
    """A3: 'dwell' renk modu, 'urgency' modundan farklı renk değerleri üretir."""
    layout = _layout()
    state = WarehouseState(layout)
    # İki bobin: farklı aciliyet AMA farklı bekleme süreleri.
    c1 = _coil("C1", 20.0, urgency=0.9); c1.stored_at = 0.0   # uzun bekledi
    c2 = _coil("C2", 12.0, urgency=0.1); c2.stored_at = 4.0   # yeni geldi
    state.place(c1, SlotCoord(0, 0, 0))
    state.place(c2, SlotCoord(1, 0, 0))

    fu = render_warehouse(state, layout, now=5.0, color_mode="urgency")
    fd = render_warehouse(state, layout, now=5.0, color_mode="dwell")
    cu = list([t for t in fu.data if t.name == FILLED_TRACE_NAME][0].marker.color)
    cd = list([t for t in fd.data if t.name == FILLED_TRACE_NAME][0].marker.color)
    assert cu != cd  # mod değişince renk değerleri değişir
    # Bekleme modunda en uzun bekleyen (c1, dwell=5) en yüksek (1.0) olmalı.
    assert max(cd) == 1.0
    # Colorbar başlığı moda göre.
    assert fd.data[-2].marker.colorbar.title.text == "Bekleme"


def test_swap_reason_shown_as_diamond_with_rationale():
    """swap_reason'lı bobin elmas trace'inde görünür; hover'da gerekçe + B1 denklemi yazar."""
    layout = _layout()
    state = WarehouseState(layout)
    coil = _coil("MOVED", 20.0, 0.4)
    coil.swap_reason = {
        "trigger_coil": "URGENT-1", "trigger_urgency": 0.9,
        "moved_from": (0, 0, 0), "moved_to": (0, 1, 0),
        "swap_cost_m": 12.0, "alt_cost_m": 31.0,
    }
    state.place(coil, SlotCoord(0, 1, 0))

    fig = render_warehouse(state, layout, now=1.0)
    moved = [t for t in fig.data if t.name == REHANDLED_TRACE_NAME][0]
    assert len(moved.x) == 1  # swap'lı bobin elmas olarak çizildi
    txt = moved.text[0]
    assert "SWAP" in txt and "URGENT-1" in txt
    assert "12.0m" in txt and "31.0m" in txt  # B1 denkleminin iki tarafı


def test_build_legend_returns_div():
    """build_legend bir HTML bileşeni döndürür ve anahtar terimleri içerir."""
    from dash import html

    legend = build_legend()
    assert isinstance(legend, html.Div)
    # Lejant metninde renk/şekil/eksen açıklamaları geçmeli (string'e düzleştirip ara).
    flat = str(legend)
    assert "Aciliyet" in flat and "Bay" in flat and "elmas" in flat


def test_color_mapping():
    """Yüksek ve düşük aciliyetli bobinler farklı renk değerlerine eşlenir."""
    layout = _layout()
    state = WarehouseState(layout)
    state.place(_coil("LOW", 20.0, 0.1), SlotCoord(0, 0, 0))
    state.place(_coil("HIGH", 12.0, 0.9), SlotCoord(1, 0, 0))

    figure = render_warehouse(state, layout)
    filled = [t for t in figure.data if t.name == FILLED_TRACE_NAME][0]
    colors = set(filled.marker.color)
    assert 0.1 in colors and 0.9 in colors
    assert len(colors) == 2  # iki farklı aciliyet -> iki farklı renk değeri


def _tiny_scenario() -> Scenario:
    """Bellekte küçük, tutarlı bir senaryo kurar (2 sipariş + araçlar)."""
    layout = _layout()
    coils = {
        "C1": _coil("C1", 20.0, order_id="O1"),
        "C2": _coil("C2", 15.0, order_id="O1"),
        "C3": _coil("C3", 22.0, order_id="O2"),
        "C4": _coil("C4", 16.0, order_id="O2"),
        "C5": _coil("C5", 18.0, order_id=None),
    }
    orders = [
        Order("O1", "V1", ["C1", "C2"], _PROD + timedelta(days=3), OrderPriority.NORMAL, OrderStatus.OPEN),
        Order("O2", "V2", ["C3", "C4"], _PROD + timedelta(days=3), OrderPriority.HIGH, OrderStatus.OPEN),
    ]

    def veh(vid, line):
        return Vehicle(
            vid, VehicleType.TRUCK, 25.0, _PROD + timedelta(hours=10), _PROD + timedelta(hours=10),
            0.0, "CARR-01", 0.8, Weather.CLEAR, 300.0, 0.3, line,
        )

    vehicles = {"V1": veh("V1", LogisticsLine.SHIP_1), "V2": veh("V2", LogisticsLine.TRAIN_A)}
    initial = [("C5", SlotCoord(1, 1, 0))]
    return Scenario(
        coils=coils, vehicles=vehicles, orders=orders, layout=layout, initial_placements=initial
    )


def test_controller_step():
    """Adım komutu simülatörü ilerletir (yerleştirme sayısı artar)."""
    controller = DashboardController(_tiny_scenario(), delay_model=None, horizon_hours=48.0)
    assert controller.sim.metrics.n_placements == 0
    controller.run_steps(20)
    assert controller.sim.metrics.n_placements > 0


def test_controller_peak_and_reset():
    """Peak bekleyen kuyruğu doldurur; reset başlangıç durumuna döndürür (initial ile)."""
    # start_empty=False: senaryonun başlangıç yerleşimini (C5) test et.
    controller = DashboardController(_tiny_scenario(), delay_model=None, horizon_hours=48.0,
                                     start_empty=False)
    controller.trigger_peak(2)
    assert len(controller.sim._pending) > 0  # zirve sipariş(ler)i bobin akıttı

    controller.run_steps(10)
    controller.reset()
    assert controller.sim.metrics.n_placements == 0
    assert controller.sim.state.occupied_count() == 1  # yalnızca initial_state (C5)


def test_controller_start_empty():
    """start_empty=True (varsayılan): depo BOŞ başlar (0 doluluk); False: initial yerleşim."""
    empty = DashboardController(_tiny_scenario(), delay_model=None, horizon_hours=48.0)
    assert empty.sim.state.occupied_count() == 0  # boştan başlar
    with_init = DashboardController(_tiny_scenario(), delay_model=None, horizon_hours=48.0,
                                    start_empty=False)
    assert with_init.sim.state.occupied_count() == 1  # initial_state (C5)


def test_metric_rows_content():
    """metric_rows beklenen etiketleri ve ML MAE satırını üretir."""
    rows = metric_rows(SimulationMetrics(rehandling_count=3, n_dispatches=2,
                                         total_loading_time_min=20.0),
                       "Heuristic", clock_hours=5.0, fill_ratio=0.5, delay_mae=6.9)
    labels = [label for label, _ in rows]
    assert "Politika" in labels and "Rehandling" in labels and "Gecikme MAE" in labels
    values = dict(rows)
    assert values["Rehandling"] == "3"
    assert values["Doluluk"] == "50%"


# --------------------------------------------------------------- P2: D2 / C3 / B2 / C1
def test_saturation_warning_in_metric_panel():
    """build_metric_panel: doluluk eşiği aşınca uyarı bandı (D2); altında nötr."""
    from src.dashboard.panels import SATURATION_THRESHOLD, build_metric_panel

    rows = metric_rows(SimulationMetrics(), "Heuristic", clock_hours=1.0, fill_ratio=0.9)
    high = str(build_metric_panel(rows, fill_ratio=0.9))
    assert "DOYUMA YAKIN" in high
    low = str(build_metric_panel(rows, fill_ratio=SATURATION_THRESHOLD - 0.1))
    assert "DOYUMA YAKIN" not in low
    # fill_ratio verilmezse uyarı yok (geriye uyumluluk).
    assert "DOYUMA YAKIN" not in str(build_metric_panel(rows))


def test_approaches_panel_lists_policies():
    """build_approaches_panel (C3) dört politikanın adını içerir."""
    from src.dashboard.panels import build_approaches_panel

    flat = str(build_approaches_panel())
    for name in ("Random", "Heuristic", "MLHeuristic", "PPO"):
        assert name in flat


def test_decision_log_records_and_renders():
    """Controller yapısal karar kaydı tutar (B2); build_decision_log onları gösterir."""
    from src.dashboard.panels import build_decision_log

    controller = DashboardController(_tiny_scenario(), delay_model=None, horizon_hours=48.0)
    controller.run_steps(12)
    assert len(controller.decisions) > 0
    d = controller.decisions[-1]
    assert {"t", "coil", "zone", "bay", "layer", "reason"} <= set(d)
    # Render: en yeni karar üstte, bobin kimliği görünür.
    flat = str(build_decision_log(controller.decisions))
    assert controller.decisions[-1]["coil"] in flat
    # Boş günlük güvenli.
    assert "henüz karar yok" in str(build_decision_log([]))


def test_comparison_controller_pairs_and_syncs():
    """ComparisonController (C1): iki şerit senkron ilerler; politika değişimi t=0'a alır."""
    from src.dashboard.controllers import ComparisonController

    cmp = ComparisonController(_tiny_scenario(), delay_model=None, horizon_hours=48.0)
    cmp.set_policy_a("Random")
    cmp.set_policy_b("Heuristic")
    cmp.run_steps(15)
    # Aynı tohumlu senaryo + senkron adım -> aynı yerleştirme sayısı (paired).
    assert cmp.lane_a.sim.metrics.n_placements == cmp.lane_b.sim.metrics.n_placements > 0
    # Metrik sözlüğü kıyas tablosu için tam.
    ma = cmp.lane_metrics(cmp.lane_a)
    assert ma["name"] == "Random" and {"rehandling", "crane_m", "loading_min", "fill"} <= set(ma)
    # Politika değişimi her iki şeridi sıfırlar (hizalı kıyas).
    cmp.set_policy_a("Heuristic")
    assert cmp.lane_a.sim.metrics.n_placements == 0
    assert cmp.lane_b.sim.metrics.n_placements == 0


def test_comparison_metrics_highlights_better():
    """build_comparison_metrics daha iyi (düşük) değeri vurgular; delta sütunu içerir."""
    from src.dashboard.panels import build_comparison_metrics

    a = {"name": "Heuristic", "rehandling": 9, "crane_m": 27000.0, "loading_min": 640.0, "fill": 0.5}
    b = {"name": "PPO", "rehandling": 3, "crane_m": 32000.0, "loading_min": 810.0, "fill": 0.5}
    flat = str(build_comparison_metrics(a, b))
    assert "Heuristic" in flat and "PPO" in flat
    assert "Rehandling" in flat and "Vinç mesafesi" in flat
    assert "Δ (A−B)" in flat  # delta sütunu (#4)


# ----------------------------------------------- dashboard ufak iyileştirmeleri
def test_metric_rows_has_rehandling_rate():
    """metric_rows 'Rehandling/saat' oranını içerir (#3); doğru hesaplanır."""
    rows = dict(metric_rows(SimulationMetrics(rehandling_count=6), "PPO",
                            clock_hours=3.0, fill_ratio=0.5))
    assert "Rehandling/saat" in rows
    assert rows["Rehandling/saat"] == "2.00"  # 6/3


def test_decision_icon_classifies():
    """_decision_icon swap/rehandling/normal kararlarını ayırt eder (#2)."""
    from src.dashboard.panels import _decision_icon

    swap_icon, swap_col = _decision_icon("yerleştirme · SWAP: COIL-X (1,2,0)→(3,4,0)")
    reh_icon, _ = _decision_icon("+2 rehandling")
    norm_icon, _ = _decision_icon("yerleştirme")
    assert swap_icon == "◆" and swap_col == "#e67e22"
    assert reh_icon == "⬆"
    assert norm_icon == "▪"
    # "+0 rehandling" rehandling SAYILMAZ (normal).
    assert _decision_icon("+0 rehandling")[0] == "▪"


def test_data_exploration_builds():
    """build_data_exploration senaryo verisinden grafik Div'i üretir (gecikme örüntüleri)."""
    from src.dashboard.panels import build_data_exploration
    from dash import html

    div = build_data_exploration(_tiny_scenario())
    assert isinstance(div, html.Div)
    flat = str(div)
    assert "Gecikme" in flat  # gecikme grafikleri var


def test_delay_panel_predicted_vs_actual():
    """build_delay_panel ML tahmini + gerçek gecikmeyi gösterir; boş güvenli."""
    from src.dashboard.panels import build_delay_panel

    delays = [{"order": "O1", "vehicle": "V1", "line": "SHIP_1", "predicted": 30.0, "actual": 75.0}]
    flat = str(build_delay_panel(delays))
    assert "O1" in flat and "30" in flat and "75" in flat and "⚠" in flat  # büyük gecikme uyarısı
    assert "aktif sevkiyat yok" in str(build_delay_panel([]))


def test_controller_active_delays():
    """controller.active_delays aktif siparişlerin araç gecikmelerini (tahmin+gerçek) döndürür."""
    controller = DashboardController(_tiny_scenario(), delay_model=None, horizon_hours=48.0)
    controller.trigger_peak(2)  # sipariş(ler)i etkinleştir
    delays = controller.active_delays()
    assert isinstance(delays, list)
    if delays:
        d = delays[0]
        assert {"order", "vehicle", "line", "predicted", "actual"} <= set(d)


def test_heldout_badge():
    """build_heldout_badge seçili politikanın held-out değerini gösterir, en iyiyi işaretler (#1)."""
    from src.dashboard.panels import build_heldout_badge

    heldout = {"Random": 32.8, "Heuristic": 8.5, "PPO": 6.83}
    flat_ppo = str(build_heldout_badge("PPO", heldout))
    assert "6.83" in flat_ppo and "en iyi" in flat_ppo  # PPO en düşük -> en iyi
    flat_heur = str(build_heldout_badge("Heuristic", heldout))
    assert "8.5" in flat_heur and "en iyi" not in flat_heur
    # Veri yoksa / bilinmeyen politika -> boş.
    assert build_heldout_badge("PPO", {}).children is None
    assert build_heldout_badge("Yok", heldout).children is None
