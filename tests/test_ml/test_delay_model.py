"""Aşama 3 gecikme tahmin modelinin kabul kriterleri testleri (docs/04 §8-§9).

Testler kendi kendine yeter: girdi araç tablosu, veri üreticisiyle bellekte
sentezlenir (data/ klasörüne bağımlı değildir). Model bir kez (modül kapsamında)
eğitilir; sabit seed determinizmi sağlar.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from src.data.config import GeneratorConfig
from src.data.vehicle_generator import generate_vehicles
from src.domain import LogisticsLine, Vehicle, VehicleType, Weather
from src.ml.delay_model import DelayPredictor
from src.ml.features import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    build_feature_matrix,
    vehicle_to_frame,
)

# Testler için sentetik araç tablosu (deterministik, orta boy).
_CONFIG = GeneratorConfig(n_vehicles=1500, seed=42)


@pytest.fixture(scope="module")
def vehicles_df():
    """Bellekte üretilmiş sentetik araç tablosu (delay_minutes dahil)."""
    return generate_vehicles(_CONFIG)


@pytest.fixture(scope="module")
def predictor(vehicles_df):
    """Sentetik tablo üzerinde bir kez eğitilmiş DelayPredictor."""
    dp = DelayPredictor(random_state=42)
    dp.train(vehicles_df)
    return dp


def test_no_leakage(vehicles_df):
    """Özellik matrisi hedef/sızıntı sütunlarını (actual_arrival, delay_minutes) içermez."""
    X, y = build_feature_matrix(vehicles_df)
    assert not (LEAKAGE_COLUMNS & set(X.columns))
    assert y is not None  # eğitim verisinde hedef bulunur


def test_feature_shape(vehicles_df):
    """One-hot sonrası beklenen 12 sütun, sabit sırada mevcut."""
    X, _ = build_feature_matrix(vehicles_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape == (len(vehicles_df), 12)
    # One-hot grupları her satırda tam olarak bir kategoriye sahip.
    assert (X[["weather_CLEAR", "weather_RAIN", "weather_SNOW"]].sum(axis=1) == 1).all()
    assert (
        X[["vehicle_type_TRUCK", "vehicle_type_TRAIN", "vehicle_type_SHIP"]].sum(axis=1) == 1
    ).all()


def test_predict_range(predictor, vehicles_df):
    """Tahminler negatif değil ve mertebe olarak makul."""
    preds = predictor.predict_batch(vehicles_df.head(200))
    assert (preds >= 0).all()
    assert preds.max() < 1000.0  # dakika cinsinden makul üst sınır
    assert preds.mean() > 0.0

    # Tekil tahmin de negatif olmayan makul bir değer döndürür.
    vehicle = Vehicle(
        vehicle_id="VEH-X", vehicle_type=VehicleType.TRUCK, max_weight_capacity_ton=25.0,
        planned_arrival=datetime(2025, 6, 1, 14, 0), actual_arrival=datetime(2025, 6, 1, 14, 0),
        delay_minutes=0.0, carrier_id="CARR-01", carrier_quality_score=0.8,
        weather=Weather.CLEAR, distance_km=300.0, traffic_index=0.3,
        target_logistics_line=LogisticsLine.TRUCK_DOCK,
    )
    pred = predictor.predict(vehicle)
    assert pred >= 0.0 and pred < 1000.0


def test_save_load(predictor, vehicles_df, tmp_path):
    """Kaydedilip yüklenen model birebir aynı tahmini üretir."""
    path = tmp_path / "delay_model.txt"
    predictor.save(str(path))
    loaded = DelayPredictor.load(str(path))
    sample = vehicles_df.head(100)
    assert np.allclose(predictor.predict_batch(sample), loaded.predict_batch(sample))


def test_importance_sanity(predictor):
    """En önemli özellikler arasında weather/carrier/distance türevleri yer alır;
    sinyalsiz zaman özellikleri (saat/gün/ay) gerçek sinyallerden daha az önemli."""
    imp = predictor.feature_importances()
    ranked = sorted(imp, key=imp.get, reverse=True)
    top6 = set(ranked[:6])

    assert "distance_km" in top6
    assert "carrier_quality_score" in top6
    assert any(w in top6 for w in ["weather_CLEAR", "weather_RAIN", "weather_SNOW"])

    # Zaman özellikleri (veri üreticisinde sinyal yok) gerçek sürücülerden zayıf olmalı.
    max_time = max(imp["planned_hour"], imp["weekday"], imp["month"])
    for signal in ["distance_km", "carrier_quality_score", "weather_CLEAR"]:
        assert imp[signal] > max_time


def test_determinism(vehicles_df):
    """Sabit random_state ile iki eğitim aynı tahminleri üretir."""
    a = DelayPredictor(random_state=42)
    a.train(vehicles_df)
    b = DelayPredictor(random_state=42)
    b.train(vehicles_df)
    sample = vehicles_df.head(100)
    assert np.allclose(a.predict_batch(sample), b.predict_batch(sample))
