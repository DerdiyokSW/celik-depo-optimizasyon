"""Bir politikayı senaryolarda koşturup metrik toplayan değerlendirme koşucusu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.simulation.event_generator import EventGenerator
from src.simulation.loaders import Scenario
from src.simulation.simulator import WarehouseSimulator

from .scenario import ScenarioSpec

if TYPE_CHECKING:
    from src.policies.base import PlacementPolicy


def build_simulator(scenario: Scenario, spec: ScenarioSpec) -> WarehouseSimulator:
    """Bir senaryo özelliğinden deterministik bir simülatör kurar."""
    return WarehouseSimulator(
        scenario.coils, scenario.orders, scenario.layout, scenario.initial_placements,
        EventGenerator(spec.event_rate_per_hour, seed=spec.event_seed),
        seed=spec.sim_seed, horizon_hours=spec.horizon_hours, vehicles=scenario.vehicles,
    )


def evaluate_policy(
    policy: "PlacementPolicy",
    scenario: Scenario,
    specs: list[ScenarioSpec],
) -> pd.DataFrame:
    """Bir politikayı tüm senaryolarda koşturur; senaryo başına metrik satırı toplar.

    Dönüş: her senaryo için bir satır içeren DataFrame (rehandling, vinç mesafesi,
    yükleme süresi, doluluk, yerleştirme/sevkiyat sayıları, ort. karar süresi ms).
    """
    rows: list[dict] = []
    for spec in specs:
        sim = build_simulator(scenario, spec)
        metrics = sim.run(policy, spec.horizon_hours)
        decision_ms = (
            sum(metrics.decision_times_ms) / len(metrics.decision_times_ms)
            if metrics.decision_times_ms
            else 0.0
        )
        rows.append(
            {
                "event_seed": spec.event_seed,
                "rehandling": metrics.rehandling_count,
                "crane_distance_m": metrics.total_crane_distance_m,
                "loading_time_min": metrics.total_loading_time_min,
                "fill_ratio": metrics.final_fill_ratio,
                "n_placements": metrics.n_placements,
                "n_dispatches": metrics.n_dispatches,
                "decision_ms": decision_ms,
            }
        )
    return pd.DataFrame(rows)


def evaluate_policy_on_pool(
    policy: "PlacementPolicy",
    scenarios: list[Scenario],
    base_seed: int = 2000,
    event_rate_per_hour: float = 12.0,
    horizon_hours: float = 24.0,
) -> pd.DataFrame:
    """Bir politikayı bir senaryo HAVUZUNDAKİ her popülasyonda koşturur (held-out eval).

    Her popülasyona benzersiz ama deterministik bir olay tohumu (``base_seed + idx``)
    verilir; aynı tohumlar tüm politikalara uygulandığında karşılaştırma eşleştirilmiş
    (paired) olur. Dönüş: popülasyon başına bir satır (overfitting ölçen GÖRÜLMEMİŞ
    veri değerlendirmesinin temeli). ``population`` sütunu havuz indeksini taşır.
    """
    rows: list[dict] = []
    for idx, scenario in enumerate(scenarios):
        spec = ScenarioSpec(
            event_seed=base_seed + idx, sim_seed=base_seed + idx,
            event_rate_per_hour=event_rate_per_hour, horizon_hours=horizon_hours,
        )
        df = evaluate_policy(policy, scenario, [spec])
        row = df.iloc[0].to_dict()
        row["population"] = idx
        rows.append(row)
    return pd.DataFrame(rows)
