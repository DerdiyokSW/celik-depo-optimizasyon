"""Sevkiyat aracı modeli (``Vehicle``).

docs/01 §5 ile birebir uyumludur. Bu nesne gecikme tahmin modelinin (Aşama 3)
ana veri kaynağıdır: ``delay_minutes`` modelin hedef değişkeni, geri kalan
alanların çoğu (hava, firma sicili, mesafe, trafik, tip) ise girdi öznitelikleridir.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .enums import LogisticsLine, VehicleType, Weather


@dataclass
class Vehicle:
    """Bobinleri depodan alan tek bir sevkiyat aracı (TIR/tren/gemi).

    Alanlar:
        vehicle_id: Benzersiz kimlik.
        vehicle_type: TRUCK / TRAIN / SHIP.
        max_weight_capacity_ton: Taşıma kapasitesi (ton).
        planned_arrival: Planlanan varış (ETA).
        actual_arrival: Gerçekleşen varış (planned + gecikme).
        delay_minutes: actual - planned, dakika. ML hedef değişkeni; her zaman ≥ 0.
        carrier_id: Lojistik firma kimliği.
        carrier_quality_score: 0..1 firma sicili (yüksek = güvenilir = az gecikme).
        weather: Varış günü hava durumu.
        distance_km: Kat edilen mesafe (50–1200).
        traffic_index: 0..1 yol yoğunluğu.
        target_logistics_line: Hizmet ettiği sevkiyat hattı (zone affinity için).
    """

    vehicle_id: str
    vehicle_type: VehicleType
    max_weight_capacity_ton: float
    planned_arrival: datetime
    actual_arrival: datetime
    delay_minutes: float
    carrier_id: str
    carrier_quality_score: float
    weather: Weather
    distance_km: float
    traffic_index: float
    target_logistics_line: LogisticsLine
