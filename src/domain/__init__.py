"""Veri sözleşmesinin (docs/01) tüm alan modelleri ve enum'ları.

Bu paket sistemin "ortak dili"dir: veri üretici, simülasyon, ML ve RL katmanları
buradaki aynı tipleri paylaşır. Pratik içe aktarma için tüm kamuya açık adlar
burada yeniden dışa verilir; böylece ``from src.domain import SteelCoil`` yeterlidir.
"""

from __future__ import annotations

from .coil import COIL_TYPE_SPECS, CoilTypeSpec, SteelCoil, max_stack_layer_for
from .enums import (
    CoilStatus,
    CoilType,
    EventType,
    LogisticsLine,
    OrderPriority,
    OrderStatus,
    QualityClass,
    VehicleType,
    Weather,
)
from .event import Event
from .order import Order
from .vehicle import Vehicle
from .warehouse import SlotCoord, WarehouseLayout

__all__ = [
    # enums
    "CoilType",
    "QualityClass",
    "CoilStatus",
    "VehicleType",
    "Weather",
    "OrderPriority",
    "OrderStatus",
    "LogisticsLine",
    "EventType",
    # modeller
    "SteelCoil",
    "SlotCoord",
    "WarehouseLayout",
    "Vehicle",
    "Order",
    "Event",
    # bobin yardımcıları
    "CoilTypeSpec",
    "COIL_TYPE_SPECS",
    "max_stack_layer_for",
]
