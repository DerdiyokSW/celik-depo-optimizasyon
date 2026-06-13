"""Aşama 7 değerlendirme hattının kabul kriterleri testleri (docs/08 §11).

Senaryo bellekte küçük üretilip geçici dizine yazılır (data/'dan bağımsız).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluation.compare import _safe_wilcoxon
from src.evaluation.runner import evaluate_policy
from src.evaluation.scenario import make_scenarios
from src.policies import HeuristicPolicy, RandomPolicy
from src.simulation.loaders import Scenario

_METRIC_COLUMNS = [
    "rehandling", "crane_distance_m", "loading_time_min", "fill_ratio",
    "n_placements", "n_dispatches", "decision_ms",
]


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    """Küçük veri setini bellekte üretip geçici dizine yazar, Scenario yükler."""
    from src.data.config import GeneratorConfig
    from src.data.generate_all import build_dataset

    coils, vehicles, orders, layout, initial = build_dataset(
        GeneratorConfig(n_coils=400, n_orders=120, n_vehicles=300, seed=42)
    )
    d = tmp_path_factory.mktemp("eval_scn")
    coils.to_parquet(d / "coils.parquet", index=False)
    vehicles.to_parquet(d / "vehicles_12m.parquet", index=False)
    orders.to_parquet(d / "orders.parquet", index=False)
    (d / "warehouse_config.json").write_text(json.dumps(layout), encoding="utf-8")
    (d / "initial_state.json").write_text(json.dumps(initial), encoding="utf-8")
    return Scenario.from_data_dir(d)


def test_scenarios_reproducible():
    """Aynı base_seed aynı senaryo kümesini verir; farklı seed farklı küme."""
    assert make_scenarios(10, base_seed=1000) == make_scenarios(10, base_seed=1000)
    assert make_scenarios(10, base_seed=1000) != make_scenarios(10, base_seed=2000)


def test_same_scenarios_all_policies(scenario):
    """Dört politika da birebir aynı senaryoları görür (adil karşılaştırma)."""
    specs = make_scenarios(5, base_seed=1000, horizon_hours=12.0)
    df_random = evaluate_policy(RandomPolicy(seed=0), scenario, specs)
    df_heur = evaluate_policy(HeuristicPolicy(), scenario, specs)
    expected = [s.event_seed for s in specs]
    assert list(df_random["event_seed"]) == expected
    assert list(df_heur["event_seed"]) == expected


def test_metrics_collected(scenario):
    """evaluate_policy her senaryo için tüm metrikleri doldurur."""
    specs = make_scenarios(3, base_seed=1000, horizon_hours=12.0)
    df = evaluate_policy(HeuristicPolicy(), scenario, specs)
    assert len(df) == 3
    for col in _METRIC_COLUMNS:
        assert col in df.columns
        assert df[col].notna().all()


def test_stat_test_runs():
    """Wilcoxon testi geçerli bir p-değeri üretir (eşleştirilmiş)."""
    a = np.array([10, 12, 8, 15, 11, 9])
    b = np.array([2, 3, 2, 4, 3, 1])
    result = _safe_wilcoxon(a, b)
    assert result["p_value"] is not None
    assert 0.0 <= result["p_value"] <= 1.0
    # Özdeş dizilerde test tanımsız -> p=1.0 ve not döner.
    same = _safe_wilcoxon(a, a.copy())
    assert same["p_value"] == 1.0
