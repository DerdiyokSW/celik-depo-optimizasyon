"""12 aylık geçmiş araç kaydını ÖĞRENİLEBİLİR gecikme örüntüsüyle üreten modül.

Bu aşamanın kalbi ``compute_delay_minutes``tir: gecikme rastgele değildir,
hava/firma sicili/mesafe/trafik/araç tipine bağlı doğrusal olmayan bir formülle
hesaplanır (docs/02 §5). Aşama 3'teki tahmin modeli tam olarak bu gizli ilişkiyi
girdi-çıktı çiftlerinden öğrenmeye çalışacaktır. ``noise`` terimi mükemmel
tahmini imkânsız kılar (gerçekçi).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain import LogisticsLine, VehicleType, Weather

from .config import HISTORY_START, GeneratorConfig

# --- Araç tipi dağılımı (Gemlik liman tesisi profili: TIR ağırlıklı, gemi belirgin) ---
VEHICLE_TYPE_DISTRIBUTION: dict[str, float] = {
    VehicleType.TRUCK.value: 0.55,
    VehicleType.SHIP.value: 0.30,
    VehicleType.TRAIN.value: 0.15,
}

# Araç tipine göre kapasite aralığı (ton). docs/01 §5: TIR~25, tren~60, gemi~120.
CAPACITY_RANGE_BY_TYPE: dict[str, tuple[float, float]] = {
    VehicleType.TRUCK.value: (22.0, 26.0),
    VehicleType.TRAIN.value: (55.0, 65.0),
    VehicleType.SHIP.value: (110.0, 130.0),
}

# Araç tipinin hizmet ettiği sevkiyat hatları (zone affinity için tutarlı eşleme).
LINES_BY_VEHICLE_TYPE: dict[str, list[str]] = {
    VehicleType.TRUCK.value: [LogisticsLine.TRUCK_DOCK.value],
    VehicleType.TRAIN.value: [LogisticsLine.TRAIN_A.value],
    VehicleType.SHIP.value: [LogisticsLine.SHIP_1.value, LogisticsLine.SHIP_2.value],
}

# Hava durumu dağılımı (yıl geneli; açık baskın, kar nadir).
WEATHER_DISTRIBUTION: dict[str, float] = {
    Weather.CLEAR.value: 0.70,
    Weather.RAIN.value: 0.22,
    Weather.SNOW.value: 0.08,
}

# Lojistik firma havuzu büyüklüğü. Her firmanın sabit bir sicil skoru olur ki
# model "şu firma sürekli geç kalıyor" örüntüsünü yakalayabilsin.
N_CARRIERS: int = 12

# ----- Gecikme formülü sabitleri (docs/02 §5). Adlandırılmış sabitler: hem -----
# ----- okunaklı hem ayarlanabilir; formül koda gömülü sihirli sayı değildir. -----
BASE_DELAY_MIN: float = 5.0                       # sabit taban gecikme
WEATHER_DELAY_MAP: dict[Weather, float] = {       # hava etkisi (dakika)
    Weather.CLEAR: 0.0,
    Weather.RAIN: 25.0,
    Weather.SNOW: 60.0,
}
CARRIER_DELAY_COEF: float = 90.0                  # düşük sicil -> 0..90 dk
DISTANCE_DELAY_COEF: float = 0.04                 # mesafe başına dk (km)
TRAFFIC_DELAY_COEF: float = 40.0                  # trafik etkisi (0..40 dk)
INTERACTION_COEF: float = 0.0008                  # mesafe×hava doğrusal-olmayan etkileşim
TYPE_DELAY_MULTIPLIER: dict[VehicleType, float] = {  # tipe göre oynaklık çarpanı
    VehicleType.TRUCK: 1.0,
    VehicleType.TRAIN: 0.6,                        # tren en stabil
    VehicleType.SHIP: 1.4,                         # gemi en oynak
}
NOISE_STD_MIN: float = 8.0                        # Gaussian gürültü std (dakika)


def compute_delay_minutes(
    weather: Weather,
    carrier_quality_score: float,
    distance_km: float,
    traffic_index: float,
    vehicle_type: VehicleType,
    rng: np.random.Generator,
) -> float:
    """Bir aracın gecikmesini (dakika) faktörlere bağlı olarak hesaplar.

    Gizli örüntü buradadır — Aşama 3 modeli tam olarak bu ilişkiyi öğrenmeye
    çalışacaktır. Doğrusal terimlerin toplamı araç tipi çarpanıyla ölçeklenir,
    üzerine Gaussian gürültü eklenir; sonuç asla 0'ın altına düşmez.
    """
    weather_effect = WEATHER_DELAY_MAP[weather]
    # Düşük sicilli firma büyük gecikme üretir (skor 1'e yaklaştıkça etki azalır).
    carrier_effect = CARRIER_DELAY_COEF * (1.0 - carrier_quality_score)
    distance_effect = DISTANCE_DELAY_COEF * distance_km
    traffic_effect = TRAFFIC_DELAY_COEF * traffic_index
    # Doğrusal olmayan etkileşim: kötü hava + uzun mesafe birlikte daha da kötü.
    interaction = INTERACTION_COEF * distance_km * weather_effect

    linear_sum = (
        BASE_DELAY_MIN
        + weather_effect
        + carrier_effect
        + distance_effect
        + traffic_effect
        + interaction
    )
    # Tip çarpanı tüm sistematik bileşeni ölçekler; gürültü en sona eklenir.
    delay = TYPE_DELAY_MULTIPLIER[vehicle_type] * linear_sum + rng.normal(0.0, NOISE_STD_MIN)
    # Negatif gecikme yok (erken varış 0 sayılır) — docs/01 doğrulama kuralı 7.
    return float(max(0.0, delay))


def generate_vehicles(config: GeneratorConfig) -> pd.DataFrame:
    """12 aylık geçmiş araç kaydını öğrenilebilir gecikme örüntüsüyle üretir.

    Önce tüm öznitelikler vektörel olarak örneklenir; sonra her araç için
    ``compute_delay_minutes`` çağrılarak gecikme hesaplanır. Tüm rastgelelik
    ``config.seed`` tohumlu tek generator iledir (determinizm). Dönüş: Vehicle
    şemasına uygun DataFrame.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_vehicles

    # --- Araç tipleri ---
    type_values = list(VEHICLE_TYPE_DISTRIBUTION.keys())
    type_probs = list(VEHICLE_TYPE_DISTRIBUTION.values())
    vehicle_types = rng.choice(type_values, size=n, p=type_probs)

    # --- Firma havuzu: her firmaya sabit bir sicil skoru ata, sonra araçları dağıt ---
    carrier_ids_pool = [f"CARR-{i + 1:02d}" for i in range(N_CARRIERS)]
    # 0.40..0.98 arası sicil; bazı firmalar belirgin daha güvenilir/güvenilmez.
    carrier_quality_pool = rng.uniform(0.40, 0.98, size=N_CARRIERS)
    quality_by_carrier = dict(zip(carrier_ids_pool, carrier_quality_pool))
    carrier_idx = rng.integers(0, N_CARRIERS, size=n)
    carrier_ids = np.array(carrier_ids_pool)[carrier_idx]
    carrier_quality = np.array([quality_by_carrier[c] for c in carrier_ids])

    # --- Kapasite (tipe bağlı aralıktan) ---
    capacity = np.empty(n, dtype=float)
    for vtype, (cap_min, cap_max) in CAPACITY_RANGE_BY_TYPE.items():
        mask = vehicle_types == vtype
        capacity[mask] = np.round(rng.uniform(cap_min, cap_max, int(mask.sum())), 1)

    # --- Planlanan varış: pencereye yayılmış ---
    offsets_min = rng.integers(0, config.window_minutes(), size=n)
    planned_arrival = pd.to_datetime(HISTORY_START) + pd.to_timedelta(offsets_min, unit="m")

    # --- Diğer öznitelikler ---
    weather_values = rng.choice(
        list(WEATHER_DISTRIBUTION.keys()), size=n, p=list(WEATHER_DISTRIBUTION.values())
    )
    distance_km = np.round(rng.uniform(50.0, 1200.0, size=n), 1)
    traffic_index = np.round(rng.random(size=n), 3)
    # Gemiler için iki hat arasında seçim için önceden bir ikili dizi çekiyoruz.
    ship_pick = rng.integers(0, 2, size=n)

    target_lines = np.empty(n, dtype=object)
    for vtype, lines in LINES_BY_VEHICLE_TYPE.items():
        mask = vehicle_types == vtype
        if len(lines) == 1:
            target_lines[mask] = lines[0]
        else:
            # SHIP: ship_pick'e göre SHIP_1 / SHIP_2
            picks = ship_pick[mask]
            target_lines[mask] = np.where(picks == 0, lines[0], lines[1])

    # --- Gecikme: her araç için scalar formülü uygula (gürültü burada çekilir) ---
    delay_minutes = np.empty(n, dtype=float)
    for i in range(n):
        delay_minutes[i] = compute_delay_minutes(
            weather=Weather(weather_values[i]),
            carrier_quality_score=float(carrier_quality[i]),
            distance_km=float(distance_km[i]),
            traffic_index=float(traffic_index[i]),
            vehicle_type=VehicleType(vehicle_types[i]),
            rng=rng,
        )
    delay_minutes = np.round(delay_minutes, 2)
    actual_arrival = planned_arrival + pd.to_timedelta(delay_minutes, unit="m")

    vehicle_ids = [f"VEH-{i + 1:06d}" for i in range(n)]

    return pd.DataFrame(
        {
            "vehicle_id": vehicle_ids,
            "vehicle_type": vehicle_types,
            "max_weight_capacity_ton": capacity,
            "planned_arrival": planned_arrival,
            "actual_arrival": actual_arrival,
            "delay_minutes": delay_minutes,
            "carrier_id": carrier_ids,
            "carrier_quality_score": np.round(carrier_quality, 3),
            "weather": weather_values,
            "distance_km": distance_km,
            "traffic_index": traffic_index,
            "target_logistics_line": target_lines,
        }
    )
