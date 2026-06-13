"""Sentetik çelik bobin envanterini üreten modül.

Her bobinin tipi gerçekçi bir dağılımla seçilir; ağırlık, çap, genişlik ve maks
istif katı tipe göre ``COIL_TYPE_SPECS`` aralıklarından örneklenir. Üretim zamanı
geçmiş pencereye yayılır. Çıktı, SteelCoil şemasına uygun bir DataFrame'dir
(``location`` ve ``order_id`` bu aşamada boştur; sipariş atama ve yerleştirme
sonraki adımlarda yapılır).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain import COIL_TYPE_SPECS, CoilStatus, CoilType, QualityClass

from .config import HISTORY_START, GeneratorConfig

# Tesis profili: soğuk hadde ağırlıklı bir üretim varsayılır (docs/02 §6).
# Anahtarlar enum değer metinleridir; np.random.choice str-enum'ları metne
# çevirdiği için baştan metinle çalışmak tutarlılığı garanti eder.
COIL_TYPE_DISTRIBUTION: dict[str, float] = {
    CoilType.COLD_ROLLED.value: 0.45,
    CoilType.GALVANIZED.value: 0.35,
    CoilType.HOT_ROLLED.value: 0.20,
}

# Tipe göre kalite sınıfı olasılıkları (A olasılığı). docs/01 §2: galvaniz
# çoğunlukla A (hassas), sıcak hadde çoğunlukla B, soğuk hadde karışık.
PROB_QUALITY_A_BY_TYPE: dict[str, float] = {
    CoilType.HOT_ROLLED.value: 0.15,
    CoilType.COLD_ROLLED.value: 0.50,
    CoilType.GALVANIZED.value: 0.85,
}


def generate_coils(config: GeneratorConfig) -> pd.DataFrame:
    """Sentetik çelik bobin envanterini üretir.

    Tip dağılımı ``COIL_TYPE_DISTRIBUTION`` ile, fiziksel öznitelikler tipe bağlı
    ``COIL_TYPE_SPECS`` aralıklarından örneklenir. Üretim zamanı geçmiş pencereye
    düzgün yayılır. Tüm rastgelelik ``config.seed`` tohumlu tek bir generator ile
    yapılır (determinizm).

    Dönüş: SteelCoil şemasına uygun DataFrame; ``order_id`` ve ``location`` None,
    ``status`` = PENDING_PLACEMENT, ``urgency_score`` = 0.0 (sonra hesaplanır).
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_coils

    # --- Tip seçimi (metin değerleri olarak) ---
    type_values = list(COIL_TYPE_DISTRIBUTION.keys())
    type_probs = list(COIL_TYPE_DISTRIBUTION.values())
    coil_types = rng.choice(type_values, size=n, p=type_probs)

    # --- Tipe bağlı fiziksel öznitelikleri maske maske doldur ---
    weight = np.empty(n, dtype=float)
    width = np.empty(n, dtype=np.int64)
    diameter = np.empty(n, dtype=np.int64)
    max_layer = np.empty(n, dtype=np.int64)
    quality = np.empty(n, dtype=object)

    for coil_type in CoilType:
        mask = coil_types == coil_type.value
        count = int(mask.sum())
        if count == 0:
            continue
        spec = COIL_TYPE_SPECS[coil_type]
        # Ağırlık sürekli; çap/genişlik tam sayı (üst sınır dahil edilsin diye +1).
        weight[mask] = np.round(rng.uniform(spec.weight_min, spec.weight_max, count), 2)
        width[mask] = rng.integers(spec.width_min, spec.width_max + 1, count)
        diameter[mask] = rng.integers(spec.diameter_min, spec.diameter_max + 1, count)
        max_layer[mask] = spec.max_stack_layer
        # Kalite: tipe özgü A olasılığıyla; geri kalan B.
        prob_a = PROB_QUALITY_A_BY_TYPE[coil_type.value]
        is_a = rng.random(count) < prob_a
        quality[mask] = np.where(is_a, QualityClass.A.value, QualityClass.B.value)

    # --- Üretim zamanı: pencereye düzgün yayılmış dakikalar ---
    offsets_min = rng.integers(0, config.window_minutes(), size=n)
    production_time = pd.to_datetime(HISTORY_START) + pd.to_timedelta(offsets_min, unit="m")

    # --- Kimlikler ---
    coil_ids = [f"COIL-{i + 1:06d}" for i in range(n)]

    return pd.DataFrame(
        {
            "coil_id": coil_ids,
            "coil_type": coil_types,
            "weight_ton": weight,
            "width_mm": width,
            "diameter_mm": diameter,
            "quality_class": quality,
            "max_stack_layer": max_layer,
            "production_time": production_time,
            # Sipariş ataması order_generator'da, yerleştirme simülasyonda yapılır.
            "order_id": pd.Series([None] * n, dtype=object),
            "status": CoilStatus.PENDING_PLACEMENT.value,
            "location": pd.Series([None] * n, dtype=object),
            "urgency_score": 0.0,
        }
    )
