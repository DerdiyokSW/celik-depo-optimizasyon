"""Aşama 2 simülasyon çekirdeğinin kabul kriterleri testleri (docs/03 §11-§12).

Testler kendi kendine yeten küçük, elle kurulmuş senaryolar kullanır (data/
klasörüne bağımlı değildir). Determinizm sabit seed'lerle garanti edilir.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.domain import (
    CoilStatus,
    CoilType,
    LogisticsLine,
    Order,
    OrderPriority,
    OrderStatus,
    QualityClass,
    SlotCoord,
    SteelCoil,
    WarehouseLayout,
)
from src.simulation.constraints import can_place, placement_violations
from src.simulation.dispatch import dispatch_order
from src.simulation.event_generator import EventGenerator
from src.simulation.simulator import WarehouseSimulator
from src.simulation.warehouse_state import WarehouseState

_PROD = datetime(2025, 1, 1)
_DEADLINE = datetime(2025, 2, 1)


def _coil(coil_id: str, weight: float, max_layer: int = 3,
          coil_type: CoilType = CoilType.COLD_ROLLED, order_id: str | None = "ORD-1") -> SteelCoil:
    """Test için tek bir bobin üretir."""
    return SteelCoil(
        coil_id, coil_type, weight, 1000, 1400, QualityClass.B, max_layer,
        _PROD, order_id, CoilStatus.PENDING_PLACEMENT, None, 0.0,
    )


def _order(order_id: str, coil_ids: list[str]) -> Order:
    """Test için tek bir sipariş üretir."""
    return Order(order_id, "VEH-1", coil_ids, _DEADLINE, OrderPriority.NORMAL, OrderStatus.OPEN)


def _layout(zone_cap: float = 1000.0) -> WarehouseLayout:
    """Küçük test deposu: 2 zone × 2 bay × 3 layer."""
    return WarehouseLayout(
        2, 2, 3,
        {0: LogisticsLine.SHIP_1, 1: LogisticsLine.TRAIN_A},
        {0: zone_cap, 1: zone_cap},
        (0, 0),
    )


# --------------------------------------------------------------- kısıt testleri
def test_constraints():
    """Her kısıt kuralı için pozitif/negatif örnek (docs/03 §5)."""
    state = WarehouseState(_layout())
    g0, g1, g2 = SlotCoord(0, 0, 0), SlotCoord(0, 0, 1), SlotCoord(0, 0, 2)

    # Pozitif: boş zemine ağır bir bobin konabilir.
    heavy = _coil("H", 20.0)
    assert can_place(state, heavy, g0)

    # Negatif: zemin boşken 1. kata konamaz (süreklilik).
    light = _coil("L", 12.0)
    assert not can_place(state, light, g1)
    assert "istif sürekli değil (alt kat boş)" in placement_violations(state, light, g1)

    state.place(heavy, g0)
    # Negatif: dolu konuma konamaz.
    assert not can_place(state, _coil("X", 5.0), g0)
    # Pozitif: ağırın üstüne hafif konabilir.
    assert can_place(state, light, g1)
    # Negatif: ağırın üstüne daha ağır konamaz (ağırlık kuralı).
    assert not can_place(state, _coil("H2", 25.0), g1)

    state.place(light, g1)
    # Negatif: maksimum kat aşımı (HOT_ROLLED max_layer=2 -> index 2 yasak).
    hot = _coil("HOT", 10.0, max_layer=2, coil_type=CoilType.HOT_ROLLED)
    assert not can_place(state, hot, g2)

    # Negatif: zone kapasite limiti (küçük limitli depoda aşım).
    small = WarehouseState(_layout(zone_cap=25.0))
    small.place(_coil("A", 20.0), SlotCoord(0, 0, 0))
    assert not can_place(small, _coil("B", 10.0), SlotCoord(0, 1, 0))  # 20+10 > 25
    assert "zone kapasite limiti aşıldı" in placement_violations(
        small, _coil("B", 10.0), SlotCoord(0, 1, 0)
    )


def test_valid_actions_subset():
    """valid_slots çıktısının her elemanı can_place testini geçer."""
    state = WarehouseState(_layout())
    state.place(_coil("H", 20.0), SlotCoord(0, 0, 0))
    state.place(_coil("M", 15.0), SlotCoord(0, 0, 1))
    coil = _coil("C", 11.0)
    valid = state.valid_slots(coil)
    assert valid, "en az bir geçerli konum olmalı"
    assert all(can_place(state, coil, slot) for slot in valid)


# ------------------------------------------------------------ rehandling testleri
def test_rehandling_known_case():
    """Hedefin üstünde 2 engelleyici -> tam olarak 2 rehandling, 1 bobin alınır."""
    state = WarehouseState(_layout())
    state.place(_coil("T", 20.0), SlotCoord(0, 0, 0))   # hedef en altta
    state.place(_coil("B1", 15.0), SlotCoord(0, 0, 1))  # engelleyici
    state.place(_coil("B2", 12.0), SlotCoord(0, 0, 2))  # engelleyici
    result = dispatch_order(state, _order("ORD-1", ["T"]))
    assert result.rehandling_count == 2
    assert result.n_retrieved == 1
    # Engelleyiciler aynı sütuna geri istiflendi (depoda hâlâ 2 bobin var).
    assert state.occupied_count() == 2


def test_rehandling_productive_not_counted():
    """Hedefin üstündeki bobin de hedefse rehandling sayılmaz (üretken hamle)."""
    state = WarehouseState(_layout())
    state.place(_coil("T1", 20.0), SlotCoord(0, 0, 0))
    state.place(_coil("T2", 15.0), SlotCoord(0, 0, 1))
    result = dispatch_order(state, _order("ORD-1", ["T1", "T2"]))
    assert result.rehandling_count == 0
    assert result.n_retrieved == 2
    assert state.occupied_count() == 0


def test_rehandling_target_on_top():
    """Hedef en üstteyse, altındaki hedef-olmayan bobine dokunulmaz (rehandling yok)."""
    state = WarehouseState(_layout())
    base = _coil("BASE", 20.0)
    state.place(base, SlotCoord(0, 0, 0))
    state.place(_coil("TOP", 12.0), SlotCoord(0, 0, 1))
    result = dispatch_order(state, _order("ORD-1", ["TOP"]))
    assert result.rehandling_count == 0
    assert result.n_retrieved == 1
    assert state.coil_at(SlotCoord(0, 0, 0)) is base  # alttaki yerinde


# ------------------------------------------- simülatör entegrasyon / determinizm
class _SpreadPolicy:
    """Deterministik baseline: en kısa stack'i tercih eden (yatay yayan) politika.

    Yeni politika sözleşmesini (decide(coil, sim)) yapısal olarak uygular.
    """

    def decide(self, coil, sim):
        valid = sim.valid_actions()
        return min(
            valid,
            key=lambda s: (sim.state.stack_height(s.zone, s.bay), s.zone, s.bay, s.layer),
        )


def _build_scenario():
    """Tutarlı küçük senaryo: 2 sipariş + birkaç başlangıç bobini."""
    coils = {
        "C1": _coil("C1", 20.0, order_id="O1"),
        "C2": _coil("C2", 15.0, order_id="O1"),
        "C3": _coil("C3", 12.0, order_id="O1"),
        "C4": _coil("C4", 22.0, order_id="O2"),
        "C5": _coil("C5", 16.0, order_id="O2"),
        "C6": _coil("C6", 18.0, order_id=None),
        "C7": _coil("C7", 14.0, order_id=None),
    }
    orders = [
        _order("O1", ["C1", "C2", "C3"]),
        Order("O2", "VEH-2", ["C4", "C5"], _DEADLINE, OrderPriority.HIGH, OrderStatus.OPEN),
    ]
    initial = [("C6", SlotCoord(1, 1, 0)), ("C7", SlotCoord(1, 0, 0))]
    return coils, orders, _layout(), initial


def _make_sim(seed: int = 1, event_seed: int = 3):
    coils, orders, layout, initial = _build_scenario()
    eg = EventGenerator(rate_per_hour=6.0, seed=event_seed)
    return WarehouseSimulator(coils, orders, layout, initial, eg, seed=seed, horizon_hours=12.0)


def test_determinism():
    """Aynı seed + aynı politika iki koşuda birebir aynı metrikleri üretir."""
    m1 = _make_sim().run(_SpreadPolicy(), 12.0)
    m2 = _make_sim().run(_SpreadPolicy(), 12.0)
    key = lambda m: (
        m.rehandling_count, m.n_dispatches, m.n_placements,
        round(m.total_crane_distance_m, 4), round(m.total_loading_time_min, 4),
    )
    assert key(m1) == key(m2)


def test_reset():
    """reset() sonrası depo başlangıç durumuna döner ve metrikler sıfırlanır."""
    coils, orders, layout, initial = _build_scenario()
    sim = WarehouseSimulator(
        coils, orders, layout, initial, EventGenerator(6.0, seed=3),
        seed=1, horizon_hours=12.0,
    )
    assert sim.state.occupied_count() == len(initial)
    sim.run(_SpreadPolicy(), 12.0)
    sim.reset()
    assert sim.state.occupied_count() == len(initial)
    assert sim.metrics.n_placements == 0
    assert sim.metrics.rehandling_count == 0


def test_scenario_not_mutated():
    """Simülatör girdi Scenario nesnelerini mutasyona uğratmaz (deep-copy izolasyonu)."""
    coils, orders, layout, initial = _build_scenario()
    WarehouseSimulator(
        coils, orders, layout, initial, EventGenerator(6.0, seed=3),
        seed=1, horizon_hours=12.0,
    ).run(_SpreadPolicy(), 12.0)
    # Dışarıdaki coil/order nesneleri temiz kalmalı.
    assert all(c.location is None for c in coils.values())
    assert all(o.status == OrderStatus.OPEN for o in orders)


def test_valid_actions_never_invalid_in_sim():
    """Simülatörün döndürdüğü her valid_action gerçekten yerleştirilebilir olmalı."""
    sim = _make_sim()
    for _ in range(40):
        coil = sim.pending_coil()
        if coil is None:
            break
        valid = sim.valid_actions()
        assert all(can_place(sim.state, coil, slot) for slot in valid)
        if valid:
            sim.apply_placement(valid[0])
        else:
            break


def test_invalid_placement_raises():
    """Dolu bir konuma apply_placement çağrısı istisna fırlatır."""
    sim = _make_sim()
    coil = sim.pending_coil()
    assert coil is not None
    valid = sim.valid_actions()
    sim.apply_placement(valid[0])
    # Aynı (artık dolu) konuma tekrar yerleştirme denenirse hata beklenir.
    nxt = sim.pending_coil()
    if nxt is not None:
        with pytest.raises(ValueError):
            sim.apply_placement(valid[0])


# --------------------------------------------------------- olay üreteci testleri
def test_event_stream_ordered():
    """EventGenerator olayları zaman-sıralı üretir; hız olay sayısını etkiler."""
    events = list(EventGenerator(rate_per_hour=12.0, seed=7).stream(24.0))
    times = [e.timestamp for e in events]
    assert times == sorted(times)
    assert all(0 <= t < 24.0 for t in times)

    low = len(list(EventGenerator(rate_per_hour=8.0, seed=7).stream(24.0)))
    high = len(list(EventGenerator(rate_per_hour=20.0, seed=7).stream(24.0)))
    assert high > low


def test_event_stream_deterministic():
    """Aynı seed iki çağrıda birebir aynı olay dizisini verir."""
    a = [(e.timestamp, e.event_type) for e in EventGenerator(10.0, seed=5).stream(10.0)]
    b = [(e.timestamp, e.event_type) for e in EventGenerator(10.0, seed=5).stream(10.0)]
    assert a == b


# ---------------------------------------------------------------- swap / relocate
def test_relocate_moves_coil_and_counts_crane_work():
    """sim.relocate, bobini yeni slota taşır + mesafe/süre metriklerini ekler."""
    coils, orders, layout, initial = _build_scenario()
    sim = WarehouseSimulator(
        coils, orders, layout, initial, EventGenerator(6.0, seed=3),
        seed=1, horizon_hours=12.0,
    )
    # initial: C6 @ (1,1,0), C7 @ (1,0,0)
    c6 = sim._coils["C6"]
    assert c6.location == SlotCoord(1, 1, 0)
    distance_before = sim.metrics.total_crane_distance_m

    new_slot = SlotCoord(0, 0, 0)
    moved = sim.relocate(c6, new_slot)

    assert c6.location == new_slot
    assert sim.state.coil_at(SlotCoord(1, 1, 0)) is None
    assert sim.state.coil_at(new_slot) is c6
    # Mesafe ve süre metriklere eklendi.
    assert moved > 0
    assert sim.metrics.total_crane_distance_m == distance_before + moved
    assert sim.metrics.total_loading_time_min > 0


def test_stored_at_set_on_placement_and_preserved_on_relocate():
    """Bobin yerleşince stored_at atanır; relocate (swap) onu KORUR (bekleme sürmez)."""
    coils, orders, layout, initial = _build_scenario()
    sim = WarehouseSimulator(
        coils, orders, layout, initial, EventGenerator(6.0, seed=3),
        seed=1, horizon_hours=12.0,
    )
    # Başlangıç envanteri t=0'dan beri depoda sayılır.
    assert sim._coils["C6"].stored_at == 0.0

    # Bir yerleştirme adımı: bekleyen bobinin stored_at'ı o anki saate eşit olmalı.
    coil = sim.pending_coil()
    assert coil is not None
    valid = sim.valid_actions()
    sim.apply_placement(valid[0])
    assert coil.stored_at == sim.clock  # giriş anı işaretlendi
    stored_before = coil.stored_at

    # Aynı bobini relocate edersek stored_at DEĞİŞMEMELİ (depoda kalmaya devam ediyor).
    empty = next(s for s in sim.state.valid_slots(coil) if s != coil.location)
    sim.relocate(coil, empty)
    assert coil.stored_at == stored_before


def test_run_does_not_enable_reposition():
    """B3 eval-güvenliği: run() yeniden konumlandırmayı ETKİNLEŞTİRMEZ (opt-in).

    _reposition_policy yalnızca dashboard'ın set_reposition_policy çağrısıyla takılır;
    değerlendirme hattı bunu çağırmadığından eval metrikleri B3'ten etkilenmez.
    """
    sim = _make_sim()
    assert sim._reposition_policy is None
    sim.run(_SpreadPolicy(), 12.0)
    assert sim._reposition_policy is None  # run() takmadı
    # Determinizm korunur (reposition kapalı): iki koşu birebir aynı.
    a = _make_sim().run(_SpreadPolicy(), 12.0).rehandling_count
    b = _make_sim().run(_SpreadPolicy(), 12.0).rehandling_count
    assert a == b


def test_reset_clears_runtime_coil_fields():
    """reset() stored_at ve swap_reason çalışma-zamanı alanlarını temizler."""
    coils, orders, layout, initial = _build_scenario()
    sim = WarehouseSimulator(
        coils, orders, layout, initial, EventGenerator(6.0, seed=3),
        seed=1, horizon_hours=12.0,
    )
    sim.run(_SpreadPolicy(), 12.0)
    sim.reset()
    # Henüz yerleşmemiş bobinlerin stored_at'ı None, swap_reason'ı None olmalı.
    for coil in sim._coils.values():
        if coil.location is None:
            assert coil.stored_at is None
        assert coil.swap_reason is None
