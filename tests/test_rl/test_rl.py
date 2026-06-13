"""Aşama 6 RL ortamı ve ödülünün kabul kriterleri testleri (docs/07 §13-§14).

Eğitim (yavaş) test edilmez; ortam API'si, action masking, gözlem/eylem eşlemesi
ve ödül fonksiyonu (saf) test edilir. Senaryo bellekte küçük kurulur.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from gymnasium.utils.env_checker import check_env

from src.domain import (
    CoilStatus, CoilType, LogisticsLine, Order, OrderPriority, OrderStatus,
    QualityClass, SlotCoord, SteelCoil, Vehicle, VehicleType, Weather, WarehouseLayout,
)
from src.rl import reward as R
from src.rl.action_space import action_space_size, index_to_slot, slot_to_index
from src.rl.warehouse_env import EnvConfig, WarehouseEnv
from src.simulation.loaders import Scenario
from src.simulation.warehouse_state import WarehouseState

_PROD = datetime(2025, 6, 1, 12, 0)


def _layout() -> WarehouseLayout:
    """Küçük RL test deposu: 2 zone × 3 bay × 2 kat = 12 slot."""
    return WarehouseLayout(
        2, 3, 2, {0: LogisticsLine.SHIP_1, 1: LogisticsLine.TRAIN_A},
        {0: 5000.0, 1: 5000.0}, (0, 0),
    )


def _rl_scenario() -> Scenario:
    """Bellekte küçük tutarlı senaryo: 4 sipariş × 3 bobin + araçlar."""
    layout = _layout()
    coils: dict[str, SteelCoil] = {}
    orders: list[Order] = []
    vehicles: dict[str, Vehicle] = {}
    for k in range(4):
        line = LogisticsLine.SHIP_1 if k % 2 == 0 else LogisticsLine.TRAIN_A
        vid = f"V{k}"
        vehicles[vid] = Vehicle(
            vid, VehicleType.TRUCK, 25.0, _PROD + timedelta(hours=k), _PROD + timedelta(hours=k),
            float(k * 20), "CARR-01", 0.8, Weather.CLEAR, 300.0, 0.3, line,
        )
        cids = []
        for i in range(3):
            cid = f"C{k}_{i}"
            coils[cid] = SteelCoil(
                cid, CoilType.COLD_ROLLED, 20.0 - i * 4, 1000, 1400, QualityClass.B, 2,
                _PROD - timedelta(days=k), f"O{k}", CoilStatus.PENDING_PLACEMENT, None, 0.0,
            )
            cids.append(cid)
        orders.append(Order(f"O{k}", vid, cids, _PROD + timedelta(days=2), OrderPriority.NORMAL, OrderStatus.OPEN))
    initial = [("C0_0", SlotCoord(1, 2, 0)), ("C0_1", SlotCoord(1, 1, 0))]
    return Scenario(coils=coils, vehicles=vehicles, orders=orders, layout=layout, initial_placements=initial)


def _make_env(use_terminal: bool = False) -> WarehouseEnv:
    return WarehouseEnv(
        EnvConfig(
            scenario=_rl_scenario(), delay_model=None, event_rate_per_hour=12.0,
            horizon_hours=12.0, max_steps=150, use_terminal_reward=use_terminal,
        )
    )


# ----------------------------------------------------- eylem uzayı / eşleme
def test_index_slot_mapping():
    """index -> SlotCoord -> index dönüşümü kayıpsız."""
    layout = _layout()
    n = action_space_size(layout)
    assert n == 2 * 3 * 2
    for index in range(n):
        assert slot_to_index(index_to_slot(index, layout), layout) == index


# ----------------------------------------------------- gymnasium API
def test_env_api():
    """WarehouseEnv gymnasium env_checker'ı geçer."""
    check_env(_make_env(use_terminal=False), skip_render_check=True)


def test_action_mask_consistency():
    """action_masks() ile valid_actions() aynı konum kümesini gösterir."""
    env = _make_env()
    env.reset(seed=1)
    for _ in range(10):
        mask = env.action_masks()
        valid_indices = {slot_to_index(s, env.layout) for s in env._sim.valid_actions()}
        mask_indices = set(np.nonzero(mask)[0].tolist())
        # valid varsa birebir eşleşir; yoksa güvenlik ağı olarak {0} olabilir.
        assert mask_indices == valid_indices or (not valid_indices and mask_indices == {0})
        obs, _, term, trunc, _ = env.step(int(np.nonzero(mask)[0][0]))
        if term or trunc:
            break


def test_observation_shape():
    """Gözlem sözlüğü tanımlı uzaya uyar; tüm değerler [0,1] aralığında float32."""
    env = _make_env()
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    for key, value in obs.items():
        assert value.dtype == np.float32, key
        assert value.min() >= 0.0 and value.max() <= 1.0, key


# ----------------------------------------------------- ödül fonksiyonu
class _StubSim:
    """Ödül yönlendirme terimini test etmek için hafif sahte simülatör."""

    def __init__(self, layout, planned_dispatch, vehicle):
        self.layout = layout
        self.clock = 0.0
        self._planned = planned_dispatch
        self._vehicle = vehicle
        # Boş state: score_slot'un istif disiplini terimi sim.state.coil_at çağırır.
        self.state = WarehouseState(layout)

    def order_of(self, coil):
        return _ORDER

    def vehicle_of(self, order):
        return self._vehicle

    def planned_dispatch_time(self, order):
        return self._planned


_ORDER = Order("O", "V", ["C"], _PROD, OrderPriority.NORMAL, OrderStatus.OPEN)


def _coil() -> SteelCoil:
    return SteelCoil("C", CoilType.COLD_ROLLED, 15.0, 1000, 1400, QualityClass.B, 2,
                     _PROD, "O", CoilStatus.PENDING_PLACEMENT, None, 0.0)


def test_reward_no_invalid_penalty():
    """Kapıya bitişik zemin (0,0,0) yerleştirme + 0 rehandling = sıfır gerçekleşen ödül
    (maliyet 0). Sabit/fizik-ihlali cezası yoktur."""
    assert R.realized_reward(0, SlotCoord(0, 0, 0)) == 0.0


def test_reward_signs():
    """Rehandling artınca ödül azalır; daha az toplam vinç mesafesi terminal ödülü artırır;
    daha iyi yerleştirme yönlendirme ödülünü artırır."""
    s = SlotCoord(0, 3, 0)
    assert R.realized_reward(2, s) < R.realized_reward(0, s)  # daha çok rehandling = düşük
    # terminal_reward(baseline_distance, agent_distance): ajan daha az mesafe → daha iyi.
    assert R.terminal_reward(1000.0, 600.0) > R.terminal_reward(1000.0, 1400.0)

    layout = _layout()
    vehicle = Vehicle("V", VehicleType.TRUCK, 25.0, _PROD, _PROD, 0.0, "C1", 0.8,
                      Weather.CLEAR, 300.0, 0.3, LogisticsLine.SHIP_1)
    sim = _StubSim(layout, planned_dispatch=2.0, vehicle=vehicle)  # acil (urgency yüksek)
    accessible = SlotCoord(0, 0, 1)
    deep = SlotCoord(0, 2, 0)
    assert R.guidance_reward(_coil(), accessible, sim) > R.guidance_reward(_coil(), deep, sim)


def test_reward_direct_cost_door_proximity():
    """v3: DOĞRUDAN maliyet — kapıya yakın + düşük kat yerleştirme, uzak/yüksekten daha iyi.

    Yerleştirme anında 'girişten slota + slottan kapıya' maliyeti cezalanır; düşük bay
    (kapıya yakın) + düşük kat daha az maliyet = daha yüksek ödül. (Dejenere 'uzağa yayıl'
    artık doğrudan pahalı.)"""
    near = SlotCoord(0, 0, 0)    # kapıya bitişik, zemin
    far_bay = SlotCoord(0, 10, 0)  # uzak bay
    high_layer = SlotCoord(0, 0, 1)  # üst kat
    assert R.realized_reward(0, near) > R.realized_reward(0, far_bay)   # yakın bay daha iyi
    assert R.realized_reward(0, near) > R.realized_reward(0, high_layer)  # düşük kat daha iyi
    # placement_cost monotonik: bay arttıkça maliyet artar.
    assert R.placement_cost(SlotCoord(0, 2, 0)) < R.placement_cost(SlotCoord(0, 8, 0))


def test_reward_scale():
    """Tipik adım ödülü kabaca [-3, 1] aralığında (makul slot + 0-1 rehandling)."""
    for slot in (SlotCoord(0, 0, 0), SlotCoord(1, 5, 1), SlotCoord(0, 10, 0)):
        assert -3.0 <= R.realized_reward(1, slot) <= 0.0
    layout = _layout()
    vehicle = Vehicle("V", VehicleType.TRUCK, 25.0, _PROD, _PROD, 0.0, "C1", 0.8,
                      Weather.CLEAR, 300.0, 0.3, LogisticsLine.SHIP_1)
    sim = _StubSim(layout, planned_dispatch=3.0, vehicle=vehicle)
    for slot in (SlotCoord(0, 0, 0), SlotCoord(1, 2, 1), SlotCoord(0, 2, 0)):
        g = R.guidance_reward(_coil(), slot, sim)
        assert -2.0 <= g <= 2.0
