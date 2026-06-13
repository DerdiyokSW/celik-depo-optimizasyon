"""Özellik mühendisliği: ham araç tablosundan model girdisi (X) ve hedefi (y).

KRİTİK — veri sızıntısı: Model yalnızca araç YOLA ÇIKMADAN bilinebilecek
özellikleri görebilir. ``actual_arrival`` ve ``delay_minutes`` hedefin kendisidir
ve ASLA X'e dahil edilmez (savunmada en çok sorulan konu). Zaman damgasından
saat/gün/ay türetilir; kategorik alanlar (hava, araç tipi) sabit kategorilerle
one-hot kodlanır, böylece çıktı sütunları her zaman aynı sıra ve içeriktedir
(tek araçlık tahminde de eğitimdekiyle birebir uyumlu).
"""

from __future__ import annotations

import pandas as pd

from src.domain import Vehicle

# Sabit kategori değerleri — one-hot sütunlarının her zaman tam ve aynı sırada
# üretilmesini garanti eder (girdide bir kategori hiç görünmese bile sütunu olur).
WEATHER_VALUES: list[str] = ["CLEAR", "RAIN", "SNOW"]
VEHICLE_TYPE_VALUES: list[str] = ["TRUCK", "TRAIN", "SHIP"]

# Modelin gördüğü nihai özellik sütunları (sabit sıra). Eğitim, tahmin ve kayıt/
# yükleme bu sırayı paylaşır.
FEATURE_COLUMNS: list[str] = (
    ["distance_km", "carrier_quality_score", "traffic_index", "planned_hour", "weekday", "month"]
    + [f"weather_{w}" for w in WEATHER_VALUES]
    + [f"vehicle_type_{t}" for t in VEHICLE_TYPE_VALUES]
)

# Hedef değişken adı.
TARGET_COLUMN: str = "delay_minutes"

# Sızıntı yaratacak, X'e asla girmemesi gereken sütunlar (denetim için).
LEAKAGE_COLUMNS: set[str] = {"actual_arrival", "delay_minutes"}


def build_feature_matrix(vehicles: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Ham araç tablosundan model girdisi (X) ve hedefi (y) üretir.

    Zaman damgasından saat/gün/ay çıkarır, kategorik alanları sabit kategorilerle
    one-hot kodlar. ``actual_arrival`` ve ``delay_minutes`` X'e ASLA dahil edilmez.
    ``delay_minutes`` tabloda yoksa (canlı tahmin) y olarak None döner.

    Dönüş: (X [FEATURE_COLUMNS sırasında], y veya None).
    """
    X = pd.DataFrame(index=vehicles.index)

    # Doğrudan sayısal özellikler.
    X["distance_km"] = vehicles["distance_km"].astype(float)
    X["carrier_quality_score"] = vehicles["carrier_quality_score"].astype(float)
    X["traffic_index"] = vehicles["traffic_index"].astype(float)

    # Zaman damgasından türetilen özellikler (saat/gün/ay).
    planned = pd.to_datetime(vehicles["planned_arrival"])
    X["planned_hour"] = planned.dt.hour.astype(int)
    X["weekday"] = planned.dt.weekday.astype(int)
    X["month"] = planned.dt.month.astype(int)

    # Sabit kategorili one-hot kodlama.
    for w in WEATHER_VALUES:
        X[f"weather_{w}"] = (vehicles["weather"] == w).astype(int)
    for t in VEHICLE_TYPE_VALUES:
        X[f"vehicle_type_{t}"] = (vehicles["vehicle_type"] == t).astype(int)

    # Sabit sütun sırasını dayat.
    X = X[FEATURE_COLUMNS]

    # Hedef yalnızca eğitim verisinde vardır.
    y = vehicles[TARGET_COLUMN].astype(float) if TARGET_COLUMN in vehicles.columns else None
    return X, y


def vehicle_to_frame(vehicle: Vehicle) -> pd.DataFrame:
    """Tek bir ``Vehicle`` nesnesini, ``build_feature_matrix``in işleyebileceği
    tek satırlık bir DataFrame'e çevirir (canlı/tekil tahmin için).

    Enum alanları metin değerlerine indirgenir; hedef/sızıntı alanları konmaz.
    """
    return pd.DataFrame(
        [
            {
                "distance_km": vehicle.distance_km,
                "carrier_quality_score": vehicle.carrier_quality_score,
                "traffic_index": vehicle.traffic_index,
                "planned_arrival": vehicle.planned_arrival,
                "weather": vehicle.weather.value,
                "vehicle_type": vehicle.vehicle_type.value,
            }
        ]
    )
