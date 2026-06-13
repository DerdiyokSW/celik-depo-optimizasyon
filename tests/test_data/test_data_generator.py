"""Aşama 1 veri üreticisinin kabul kriterleri testleri (docs/02 §7-§8).

Tüm testler küçük ama temsil edici bir veri seti üzerinde koşar (hız için);
``seed`` sabittir, böylece testler tekrarlanabilirdir. Veri seti modül kapsamlı
bir fixture ile bir kez üretilir.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.config import GeneratorConfig
from src.data.generate_all import build_dataset
from src.data.validation import validate_all
from src.domain import COIL_TYPE_SPECS, CoilType

# Hızlı ama anlamlı boyut: tüm tipler ve örüntüler gözlemlenebilir kalır.
TEST_CONFIG = GeneratorConfig(
    n_coils=600, n_orders=150, n_vehicles=300, n_months=12, seed=42
)


@pytest.fixture(scope="module")
def dataset():
    """Test veri setini bir kez üretir: (coils, vehicles, orders, layout, initial_state)."""
    return build_dataset(TEST_CONFIG)


def test_coil_ranges(dataset):
    """Her bobin tipinin ağırlık/çap/genişlik değerleri sözleşme aralıklarında olmalı."""
    coils, *_ = dataset
    for coil_type in CoilType:
        spec = COIL_TYPE_SPECS[coil_type]
        sub = coils[coils["coil_type"] == coil_type.value]
        assert not sub.empty, f"{coil_type.value} tipinde hiç bobin üretilmedi"
        assert sub["weight_ton"].between(spec.weight_min, spec.weight_max).all()
        assert sub["width_mm"].between(spec.width_min, spec.width_max).all()
        assert sub["diameter_mm"].between(spec.diameter_min, spec.diameter_max).all()
        # Tip–kat tutarlılığı (kural 8) tek tipte sabit olmalı.
        assert (sub["max_stack_layer"] == spec.max_stack_layer).all()


def test_id_uniqueness(dataset):
    """Tüm coil_id, vehicle_id, order_id değerleri benzersiz olmalı."""
    coils, vehicles, orders, *_ = dataset
    assert coils["coil_id"].is_unique
    assert vehicles["vehicle_id"].is_unique
    assert orders["order_id"].is_unique


def test_referential_integrity(dataset):
    """Bobin–sipariş–araç bağları çift yönlü tutarlı olmalı."""
    coils, vehicles, orders, *_ = dataset
    valid_vehicle_ids = set(vehicles["vehicle_id"])
    coil_to_order = dict(zip(coils["coil_id"], coils["order_id"]))

    for row in orders.itertuples():
        # Araç referansı geçerli.
        assert row.vehicle_id in valid_vehicle_ids
        for coil_id in row.coil_ids:
            # Bobin var ve order_id'si bu siparişe işaret ediyor (ileri yön).
            assert coil_id in coil_to_order
            assert coil_to_order[coil_id] == row.order_id

    # Ters yön: order_id'si dolu her bobin ilgili siparişin listesinde olmalı.
    order_to_coils = {row.order_id: set(row.coil_ids) for row in orders.itertuples()}
    for coil_id, order_id in coil_to_order.items():
        if order_id is not None:
            assert coil_id in order_to_coils[order_id]


def test_delay_non_negative(dataset):
    """Hiçbir delay_minutes negatif olmamalı (kural 7)."""
    _, vehicles, *_ = dataset
    assert (vehicles["delay_minutes"] >= 0).all()


def test_delay_pattern(dataset):
    """Gizli gecikme örüntüsü gözlemlenebilir olmalı: SNOW > RAIN > CLEAR ve
    düşük sicilli firmaların ortalama gecikmesi yüksek sicillilerden büyük."""
    _, vehicles, *_ = dataset
    mean_by_weather = vehicles.groupby("weather")["delay_minutes"].mean()
    assert mean_by_weather["SNOW"] > mean_by_weather["RAIN"] > mean_by_weather["CLEAR"]

    median_q = vehicles["carrier_quality_score"].median()
    low_q = vehicles[vehicles["carrier_quality_score"] < median_q]["delay_minutes"].mean()
    high_q = vehicles[vehicles["carrier_quality_score"] >= median_q]["delay_minutes"].mean()
    assert low_q > high_q


def test_determinism():
    """Aynı seed iki çalıştırmada birebir aynı veri setini üretmeli."""
    c1, v1, o1, l1, s1 = build_dataset(TEST_CONFIG)
    c2, v2, o2, l2, s2 = build_dataset(TEST_CONFIG)
    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(v1, v2)
    pd.testing.assert_frame_equal(o1, o2)
    assert l1 == l2
    assert s1 == s2


def test_initial_state_valid(dataset):
    """Başlangıç yerleşimi süreklilik ve ağırlık kurallarını ihlal etmemeli."""
    coils, _, _, _, initial_state = dataset
    weight_by_coil = dict(zip(coils["coil_id"], coils["weight_ton"]))

    occupied: dict[tuple[int, int, int], float] = {}
    for p in initial_state["placements"]:
        occupied[(p["zone"], p["bay"], p["layer"])] = weight_by_coil[p["coil_id"]]

    assert initial_state["n_placed"] == len(initial_state["placements"])
    assert occupied, "başlangıç durumunda hiç yerleşim yok"

    for (zone, bay, layer), weight in occupied.items():
        if layer > 0:
            below = (zone, bay, layer - 1)
            # Süreklilik: alt kat dolu olmalı.
            assert below in occupied
            # Ağırlık: üstteki bobin alttakinden hafif olmalı.
            assert weight < occupied[below]


def test_initial_state_not_optimal(dataset):
    """Başlangıç bilinçli olarak bozuk olmalı: bazı acil (eski) bobinler
    erişimi zor alt katlara gömülmüş olmalı — yani yerleşim optimumdan uzak."""
    coils, _, _, _, initial_state = dataset
    prod_by_coil = dict(zip(coils["coil_id"], coils["production_time"]))

    bottom_times = [
        prod_by_coil[p["coil_id"]] for p in initial_state["placements"] if p["layer"] == 0
    ]
    top_times = [
        prod_by_coil[p["coil_id"]] for p in initial_state["placements"] if p["layer"] > 0
    ]
    assert bottom_times and top_times
    # Vekil aciliyet = erken üretim. Zemine gömülenlerin ortalama üretim zamanı,
    # üst katlardakinden daha erken olmalı (daha acil olanlar altta = anti-optimal).
    assert pd.Series(bottom_times).mean() < pd.Series(top_times).mean()


def test_validation_catches_corruption(dataset):
    """validate_all bozuk veriyi yakalamalı: negatif gecikme enjekte edip
    istisna beklenir (doğrulama kapısının gerçekten çalıştığının kanıtı)."""
    coils, vehicles, orders, layout, initial_state = dataset
    corrupted = vehicles.copy()
    corrupted.loc[corrupted.index[0], "delay_minutes"] = -10.0
    with pytest.raises(ValueError):
        validate_all(coils, corrupted, orders, layout, initial_state)
