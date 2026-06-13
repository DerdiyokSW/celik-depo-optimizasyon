"""Dinamik olay modeli (``Event``).

docs/01 §7 ile birebir uyumludur. Olaylar simülasyon zaman ekseninde üretilir
(Aşama 2'deki event generator) ve sistemin stokastik/dinamik doğasını sağlar:
yeni sipariş, iptal, araç gecikmesi, öncelik değişimi, zirve yükü.

``payload`` olay tipine göre değişen serbest biçimli bir sözlüktür; her tipin
beklediği anahtarlar docs/01 §7'de tanımlıdır.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .enums import EventType


@dataclass
class Event:
    """Simülasyon sırasında belirli bir anda gerçekleşen tek bir dinamik olay.

    Alanlar:
        timestamp: Simülasyon saatinde olayın anı (saat cinsinden, kayan nokta).
        event_type: Olayın tipi.
        payload: Olaya özgü veri. İçeriği ``event_type``a göre değişir
            (ör. NEW_ORDER için {"order": Order}, VEHICLE_DELAY için
            {"vehicle_id": str, "extra_delay_minutes": float}).
    """

    timestamp: float
    event_type: EventType
    payload: dict[str, Any]
