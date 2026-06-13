"""Aşama 1 giriş noktası: tüm veri setini sırayla üretir, doğrular, diske yazar.

Çalıştırma: ``python -m src.data.generate_all``

Sıra önemlidir: bobinler -> araçlar -> siparişler (coils.order_id'yi günceller)
-> depo konfigürasyonu -> bozuk başlangıç durumu. Yazımdan önce 8 doğrulama
kuralı koşar; herhangi biri ihlal edilirse hiçbir dosya yazılmaz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from .coil_generator import generate_coils
from .config import GeneratorConfig
from .layout_generator import generate_initial_state, generate_warehouse_config
from .order_generator import generate_orders
from .validation import validate_all
from .vehicle_generator import generate_vehicles

# Proje kökü: bu dosya src/data/ altında olduğundan iki üst dizin köktür.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"


def build_dataset(
    config: GeneratorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """Tüm veri setini bellekte üretir ve doğrular (diske yazmadan).

    Üretim sırasını ve çift yönlü bağ kurulumunu garanti eder. Testler bu
    fonksiyonu doğrudan kullanır (dosya G/Ç olmadan hızlı doğrulama için).

    Dönüş: (coils, vehicles, orders, layout, initial_state).
    """
    coils = generate_coils(config)
    vehicles = generate_vehicles(config)
    # generate_orders, coils.order_id sütununu yerinde günceller (yan etki).
    orders = generate_orders(config, coils, vehicles)
    layout = generate_warehouse_config(config)
    initial_state = generate_initial_state(config, coils, layout)

    # Yazımdan önce tüm kuralları doğrula — ihlalde istisna fırlar, dosya yazılmaz.
    validate_all(coils, vehicles, orders, layout, initial_state)
    return coils, vehicles, orders, layout, initial_state


def write_dataset(
    coils: pd.DataFrame, vehicles: pd.DataFrame, orders: pd.DataFrame,
    layout: dict, initial_state: dict, out_dir: Path,
) -> None:
    """Üretilmiş veri setini verilen dizine yazar (data/ ile birebir aynı şema).

    Tek bir yazım noktası: hem ``main`` (varsayılan data/) hem senaryo havuzu üretici
    (``generate_pool``) bunu kullanır; böylece dosya adları/biçimleri tek yerden gelir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Tabular veriler parquet (sıkıştırılmış, tipli); konfigürasyonlar JSON.
    coils.to_parquet(out_dir / "coils.parquet", index=False)
    vehicles.to_parquet(out_dir / "vehicles_12m.parquet", index=False)
    orders.to_parquet(out_dir / "orders.parquet", index=False)
    (out_dir / "warehouse_config.json").write_text(
        json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "initial_state.json").write_text(
        json.dumps(initial_state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main(config: GeneratorConfig | None = None, out_dir: Path | None = None) -> None:
    """Tüm veri setini üretir, doğrular ve (varsayılan) data/ klasörüne yazar.

    ``out_dir`` verilirse oraya yazar (senaryo havuzu üretimi bunu kullanır).
    """
    if config is None:
        config = GeneratorConfig()
    target_dir = Path(out_dir) if out_dir is not None else DATA_DIR

    # Windows konsolu (cp1252) Türkçe karakterleri kodlayamayabilir; çıktıyı UTF-8'e al.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    coils, vehicles, orders, layout, initial_state = build_dataset(config)
    write_dataset(coils, vehicles, orders, layout, initial_state, target_dir)

    # Özet rapor (göstermelik değil — gerçekten yazılan verinin sayıları).
    print(f"Veri seti üretildi -> {target_dir}")
    print(f"  coils.parquet         : {len(coils):>6} bobin")
    print(f"  vehicles_12m.parquet  : {len(vehicles):>6} araç")
    print(f"  orders.parquet        : {len(orders):>6} sipariş")
    print(f"  warehouse_config.json : {layout['n_zones']}×{layout['n_bays']}×{layout['n_layers']} = "
          f"{layout['n_zones'] * layout['n_bays'] * layout['n_layers']} konum")
    print(f"  initial_state.json    : {initial_state['n_placed']:>6} bobin yerleştirildi")
    print("Tüm 8 doğrulama kuralı geçti.")


if __name__ == "__main__":
    main()
