"""data/ klasöründeki veri seti dosyalarını domain nesnelerine yükleyen modül.

Aşama 1'in ürettiği parquet/JSON dosyalarını okuyup ``SteelCoil``, ``Vehicle``,
``Order``, ``WarehouseLayout`` ve başlangıç yerleşimi nesnelerine çevirir.
Simülatör (ve ileride değerlendirme/RL) bu köprüyü kullanarak gerçek veriyle kurulur.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.domain import (
    CoilStatus,
    CoilType,
    LogisticsLine,
    Order,
    OrderPriority,
    OrderStatus,
    QualityClass,
    SlotCoord,
    SteelCoil,
    Vehicle,
    VehicleType,
    WarehouseLayout,
    Weather,
)

# Varsayılan veri dizini: proje kökündeki data/.
DEFAULT_DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data"


def _opt_str(value: object) -> str | None:
    """Parquet'ten gelen boş (None/NaN) değeri None'a, dolu değeri str'e çevirir."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def load_coils(path: Path) -> dict[str, SteelCoil]:
    """coils.parquet'i ``coil_id -> SteelCoil`` sözlüğüne yükler (location boş)."""
    df = pd.read_parquet(path)
    coils: dict[str, SteelCoil] = {}
    for row in df.itertuples():
        coils[row.coil_id] = SteelCoil(
            coil_id=str(row.coil_id),
            coil_type=CoilType(row.coil_type),
            weight_ton=float(row.weight_ton),
            width_mm=int(row.width_mm),
            diameter_mm=int(row.diameter_mm),
            quality_class=QualityClass(row.quality_class),
            max_stack_layer=int(row.max_stack_layer),
            production_time=row.production_time,  # pd.Timestamp (datetime alt sınıfı)
            order_id=_opt_str(row.order_id),
            status=CoilStatus(row.status),
            location=None,
            urgency_score=float(row.urgency_score),
        )
    return coils


def load_vehicles(path: Path) -> dict[str, Vehicle]:
    """vehicles_12m.parquet'i ``vehicle_id -> Vehicle`` sözlüğüne yükler."""
    df = pd.read_parquet(path)
    vehicles: dict[str, Vehicle] = {}
    for row in df.itertuples():
        vehicles[row.vehicle_id] = Vehicle(
            vehicle_id=str(row.vehicle_id),
            vehicle_type=VehicleType(row.vehicle_type),
            max_weight_capacity_ton=float(row.max_weight_capacity_ton),
            planned_arrival=row.planned_arrival,
            actual_arrival=row.actual_arrival,
            delay_minutes=float(row.delay_minutes),
            carrier_id=str(row.carrier_id),
            carrier_quality_score=float(row.carrier_quality_score),
            weather=Weather(row.weather),
            distance_km=float(row.distance_km),
            traffic_index=float(row.traffic_index),
            target_logistics_line=LogisticsLine(row.target_logistics_line),
        )
    return vehicles


def load_orders(path: Path) -> list[Order]:
    """orders.parquet'i ``Order`` listesine yükler (coil_ids liste sütunu)."""
    df = pd.read_parquet(path)
    orders: list[Order] = []
    for row in df.itertuples():
        orders.append(
            Order(
                order_id=str(row.order_id),
                vehicle_id=str(row.vehicle_id),
                coil_ids=[str(c) for c in row.coil_ids],
                deadline=row.deadline,
                priority=OrderPriority(row.priority),
                status=OrderStatus(row.status),
            )
        )
    return orders


def load_layout(path: Path) -> WarehouseLayout:
    """warehouse_config.json'u ``WarehouseLayout`` nesnesine yükler."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return WarehouseLayout.from_dict(data)


def load_initial_placements(path: Path) -> list[tuple[str, SlotCoord]]:
    """initial_state.json'u ``(coil_id, SlotCoord)`` listesine yükler."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        (p["coil_id"], SlotCoord(p["zone"], p["bay"], p["layer"]))
        for p in data["placements"]
    ]


@dataclass
class Scenario:
    """Bir simülasyonu kurmak için gereken tüm yüklenmiş veri.

    Alanlar: coils (id->SteelCoil), vehicles (id->Vehicle), orders (liste),
    layout (WarehouseLayout), initial_placements ((coil_id, SlotCoord) listesi).
    """

    coils: dict[str, SteelCoil]
    vehicles: dict[str, Vehicle]
    orders: list[Order]
    layout: WarehouseLayout
    initial_placements: list[tuple[str, SlotCoord]]

    @classmethod
    def from_data_dir(cls, data_dir: Path | None = None) -> "Scenario":
        """data/ klasöründeki 5 dosyayı okuyup eksiksiz bir Scenario kurar."""
        d = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        return cls(
            coils=load_coils(d / "coils.parquet"),
            vehicles=load_vehicles(d / "vehicles_12m.parquet"),
            orders=load_orders(d / "orders.parquet"),
            layout=load_layout(d / "warehouse_config.json"),
            initial_placements=load_initial_placements(d / "initial_state.json"),
        )


def load_pool(pool_dir: Path | str) -> list[Scenario]:
    """Bir senaryo havuzu dizinindeki tüm ``seed_*`` veri setlerini Scenario listesi olarak yükler.

    ``data/pool/train`` veya ``data/pool/test`` altındaki her ``seed_S`` alt klasörü
    bir bağımsız popülasyondur. Train/test ayrımı (held-out genelleme ölçümü) bu
    havuzlarla yapılır. Tohum sırasına göre deterministik sıralı döner.
    """
    base = Path(pool_dir)
    if not base.exists():
        raise FileNotFoundError(
            f"Havuz dizini yok: {base}. Önce 'python -m src.data.generate_pool' koş."
        )
    # seed_S klasörlerini tohum numarasına göre sırala (deterministik sıra).
    seed_dirs = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.startswith("seed_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    return [Scenario.from_data_dir(d) for d in seed_dirs]
