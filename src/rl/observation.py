"""Simülatör durumundan PPO gözlem sözlüğü üretimi.

Gözlem üç parçalıdır (docs/07 §6): depo tensörü, bekleyen bobin vektörü, küresel
göstergeler. Tüm değerler ~[0,1] aralığına normalize edilir (sinir ağı eğitimi
için şart). Bu modül gymnasium'a bağlı DEĞİLDİR (uzay tanımı warehouse_env'de);
böylece PPOPolicy de bu fonksiyonu hafifçe kullanabilir.

Hibrit mimarinin can alıcı noktası: bekleyen bobinin vektörüne, aracın
``DelayPredictor`` ile tahmin edilen gecikmesi konur — PPO da Aşama 3'ün ML
çıktısını girdi alır.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.domain import CoilType
from src.policies.scoring import coil_urgency, planned_urgency

if TYPE_CHECKING:
    from src.ml.delay_model import DelayPredictor
    from src.simulation.simulator import WarehouseSimulator

# Normalizasyon sabitleri (tipe/aralığa göre kabaca üst sınırlar).
MAX_WEIGHT_TON: float = 30.0
MAX_DELAY_MIN: float = 600.0   # tahmini gecikme normalizasyonu
MAX_QUEUE_LEN: float = 50.0    # bekleyen kuyruğu normalizasyonu

PENDING_DIM: int = 8
GLOBAL_DIM: int = 3
WAREHOUSE_CHANNELS: int = 3   # doluluk, normalize ağırlık, normalize aciliyet

# Bobin tipinin one-hot sırası (gözlemde sabit).
_TYPE_INDEX: dict[CoilType, int] = {
    CoilType.COLD_ROLLED: 0,
    CoilType.GALVANIZED: 1,
    CoilType.HOT_ROLLED: 2,
}


def warehouse_shape(layout) -> tuple[int, int, int, int]:
    """Depo tensörünün şekli: (zone, bay, layer, kanal)."""
    return (layout.n_zones, layout.n_bays, layout.n_layers, WAREHOUSE_CHANNELS)


def build_observation(sim: "WarehouseSimulator", delay_model: "DelayPredictor | None") -> dict:
    """Simülatörün anlık durumundan PPO gözlem sözlüğünü üretir.

    Dönüş: {"warehouse": (Z,B,L,3), "pending_coil": (8,), "global": (3,)} —
    tümü float32 ve [0,1] aralığına kırpılmış.
    """
    layout = sim.layout

    # --- Depo tensörü: her dolu konum için doluluk, ağırlık, aciliyet ---
    warehouse = np.zeros(warehouse_shape(layout), dtype=np.float32)
    for coil in sim.state.stored_coils():
        loc = coil.location
        if loc is None:
            continue
        warehouse[loc.zone, loc.bay, loc.layer, 0] = 1.0
        warehouse[loc.zone, loc.bay, loc.layer, 1] = min(1.0, coil.weight_ton / MAX_WEIGHT_TON)
        warehouse[loc.zone, loc.bay, loc.layer, 2] = planned_urgency(coil, sim)

    # --- Bekleyen bobin vektörü ---
    pending = np.zeros(PENDING_DIM, dtype=np.float32)
    coil = sim.pending_coil()
    if coil is not None:
        order = sim.order_of(coil)
        vehicle = sim.vehicle_of(order)
        pending[_TYPE_INDEX[coil.coil_type]] = 1.0
        pending[3] = min(1.0, coil.weight_ton / MAX_WEIGHT_TON)
        pending[4] = coil.max_stack_layer / layout.n_layers
        # Planlanan (gecikmesiz) aciliyet.
        pending[5] = planned_urgency(coil, sim)
        # Tahmini gecikme (normalize) — hibrit mimarinin ML girdisi.
        predicted_delay_min = (
            float(delay_model.predict(vehicle)) if (delay_model is not None and vehicle is not None) else 0.0
        )
        pending[6] = min(1.0, predicted_delay_min / MAX_DELAY_MIN)
        # ML ile düzeltilmiş aciliyet: planlanan dispatch + tahmini gecikme - şu an.
        planned = sim.planned_dispatch_time(order)
        if planned is not None:
            adjusted_hours = (planned + predicted_delay_min / 60.0) - sim.clock
            pending[7] = coil_urgency(max(0.0, adjusted_hours))

    # --- Küresel göstergeler ---
    global_state = np.array(
        [
            sim.state.fill_ratio(),
            min(1.0, sim.pending_count / MAX_QUEUE_LEN),
            min(1.0, sim.clock / sim.horizon) if sim.horizon > 0 else 0.0,
        ],
        dtype=np.float32,
    )

    return {"warehouse": warehouse, "pending_coil": pending, "global": global_state}
