"""Sevkiyat siparişi modeli (``Order``).

docs/01 §6 ile birebir uyumludur. Bir sipariş, tek bir araca yüklenecek bir
bobin grubudur. Bobin–sipariş bağı çift yönlüdür: siparişin ``coil_ids`` listesi
ile her bobinin ``order_id`` alanı birbirini doğrulamalıdır (docs/01 doğrulama
kuralı 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .enums import OrderPriority, OrderStatus


@dataclass
class Order:
    """Bir araca yüklenecek bobin grubunu temsil eden sevkiyat siparişi.

    Alanlar:
        order_id: Benzersiz kimlik (ör. "ORD-000001").
        vehicle_id: Siparişi karşılayan aracın kimliği.
        coil_ids: Yüklenecek bobinlerin kimlik listesi.
        deadline: Son sevk zamanı; üretim zamanından ve varış zamanından sonradır.
        priority: NORMAL / HIGH / URGENT.
        status: OPEN / IN_PROGRESS / FULFILLED / CANCELLED.
    """

    order_id: str
    vehicle_id: str
    coil_ids: list[str]
    deadline: datetime
    priority: OrderPriority
    status: OrderStatus
