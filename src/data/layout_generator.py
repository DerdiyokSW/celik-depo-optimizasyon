"""Depo konfigürasyonunu ve bilinçli olarak BOZUK başlangıç durumunu üreten modül.

``generate_warehouse_config`` deponun değişmez geometrisini (4×12×3) üretir.
``generate_initial_state`` ise optimizasyon politikalarının iyileştirme gücünü
ölçmek için referans zemin olan, fiziksel olarak GEÇERLİ ama bilinçle optimumdan
uzak bir başlangıç yerleşimi üretir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain import LogisticsLine, SlotCoord, WarehouseLayout

from .config import GeneratorConfig

# Depo geometrisi sabitleri (docs/01 §3-§4). Daha kompakt dikdörtgen hol; dikey istif 2 kat.
# Bay sayısı 36'ya küçültüldü: PPO eylem uzayını ~%45 daraltır (1056 → 576 slot),
# CNN feature extractor için daha tractable.
N_ZONES: int = 8
N_BAYS: int = 36
N_LAYERS: int = 2

# 4 lojistik hat, hat başına 2 zone (8 zone). zone 0-1 SHIP_1, 2-3 SHIP_2, ...
_LINES: list[LogisticsLine] = [
    LogisticsLine.SHIP_1,
    LogisticsLine.SHIP_2,
    LogisticsLine.TRAIN_A,
    LogisticsLine.TRUCK_DOCK,
]
ZONE_LOGISTICS: dict[int, LogisticsLine] = {z: _LINES[z // 2] for z in range(N_ZONES)}

# Zone başına toplam tonaj limiti. Bir zone'da 66×2 = 132 konum × ~17 ton ortalama
# ≈ 2240 ton olur; 3000 ton dolu bir zone'a izin verirken aşırı ağır kümelenmeyi
# sınırlar (kapasite normal işleyişte nadiren bağlar).
ZONE_MAX_WEIGHT_TON: float = 3000.0

# Başlangıçta dolu olacak konum oranı. %30: rehandling fırsatı yaratacak kadar dolu
# ama olay akışıyla hemen taşmayacak kadar bol boş alan bırakır (1056 slotun ~%30'u).
INITIAL_FILL_RATIO: float = 0.30


def generate_warehouse_config(config: GeneratorConfig) -> dict:
    """WarehouseLayout şemasına uygun depo konfigürasyonunu üretir (4×12×3).

    Rastgelelik içermez (deterministik sabit geometri). JSON'a yazılabilir saf
    sözlük döndürür.
    """
    layout = WarehouseLayout(
        n_zones=N_ZONES,
        n_bays=N_BAYS,
        n_layers=N_LAYERS,
        zone_logistics=dict(ZONE_LOGISTICS),
        zone_max_weight_ton={z: ZONE_MAX_WEIGHT_TON for z in range(N_ZONES)},
        entry_point=(0, 0),  # üretim bandı çıkışı: bay=0, zone=0 referansı
    )
    return layout.to_dict()


def generate_initial_state(
    config: GeneratorConfig,
    coils: pd.DataFrame,
    layout: dict,
) -> dict:
    """Bilinçli olarak BOZUK ama fiziksel olarak GEÇERLİ bir başlangıç yerleşimi üretir.

    Aciliyet vekili: gerçek sevkiyat aciliyeti (deadline) simülasyonda hesaplanır;
    burada erişilebilir bir vekil olarak ÜRETİM ZAMANI kullanılır — daha erken
    üretilen bobin daha uzun beklemiştir, dolayısıyla daha aciledir (FIFO mantığı).

    Bozukluk stratejisi: bobinler KOLON-KOLON 2 katlı istiflenir; her kolonun
    ALTINA daha acil (eski üretim) bir bobin, ÜSTÜNE daha az acil ve daha hafif bir
    bobin konur. Böylece acil bobin gömülü kalır — sevkiyatında üstündeki taşınmak
    zorundadır (rehandling). Kolonlar depoya dağıtılır (gerçekçi seyrek envanter).
    Bu, "erken sevkiyat -> alt kat" anti-optimal desenidir; istif kuralları İHLAL
    EDİLMEZ (süreklilik, ağırlık azalması, zone kapasitesi).

    Dönüş: ``{"placements": [{coil_id, zone, bay, layer}, ...], "n_placed": int}``.
    """
    rng = np.random.default_rng(config.seed)
    wl = WarehouseLayout.from_dict(layout)

    # Doldurulacak bobin sayısı (toplam konumun belli bir oranı).
    target_count = int(INITIAL_FILL_RATIO * wl.total_slots())

    # Aciliyet vekili = üretim zamanı (erken = uzun beklemiş = acil). En acil
    # target_count bobini başlangıçta kabul et; aciliyete göre (eskiden yeniye) sırala.
    placed = coils.sort_values("production_time").head(target_count)
    records = [(str(r.coil_id), float(r.weight_ton)) for r in placed.itertuples()]

    # Yarı yarıya böl: daha acil (eski) yarı ALTA gömülür, daha az acil yarı ÜSTE.
    half = len(records) // 2
    bottom_pool = records[:half]        # acil -> zemin (gömülü)
    top_pool = records[half:]           # daha az acil -> üst kat
    top_used = [False] * len(top_pool)

    # Stack'leri karıştır ki başlangıç envanteri depoya dağılsın (determinist).
    stacks = [(z, b) for z in range(wl.n_zones) for b in range(wl.n_bays)]
    rng.shuffle(stacks)
    zone_weight: dict[int, float] = {z: 0.0 for z in range(wl.n_zones)}
    placements: list[dict] = []

    for stack_index, (coil_id, weight) in enumerate(bottom_pool):
        if stack_index >= len(stacks):
            break
        zone, bay = stacks[stack_index]
        # Zemine acil bobini koy (kapasite uygunsa).
        if zone_weight[zone] + weight > wl.zone_max_weight_ton[zone]:
            continue
        placements.append({"coil_id": coil_id, "zone": zone, "bay": bay, "layer": 0})
        zone_weight[zone] += weight
        # Üste, alttakinden HAFİF ve daha az acil bir bobin ara (ağırlık kuralı).
        for j, (top_id, top_weight) in enumerate(top_pool):
            if top_used[j]:
                continue
            if top_weight < weight and zone_weight[zone] + top_weight <= wl.zone_max_weight_ton[zone]:
                placements.append({"coil_id": top_id, "zone": zone, "bay": bay, "layer": 1})
                zone_weight[zone] += top_weight
                top_used[j] = True
                break

    return {"placements": placements, "n_placed": len(placements)}
