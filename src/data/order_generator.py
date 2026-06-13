"""Siparişleri üreten ve her siparişe bobin + araç atayan modül.

Bobin–sipariş bağı ÇİFT YÖNLÜ kurulur: ``coils`` tablosundaki ``order_id`` da
yerinde güncellenir (yan etki). Bir sipariş tek bir araçla karşılanır; aracın
kapasitesi aşılmaz ve bir bobin yalnızca tek bir siparişe atanır. Zaman
tutarlılığı sağlanır: ``deadline`` hem bobin üretim zamanından hem de aracın
planlanan varışından sonradır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain import OrderPriority, OrderStatus

from .config import GeneratorConfig

# Sipariş başına bobin sayısı üst sınırı (kapasite çoğu zaman bundan önce bağlar).
MAX_COILS_PER_ORDER: int = 8

# Öncelik dağılımı (docs/02 §6).
PRIORITY_DISTRIBUTION: dict[str, float] = {
    OrderPriority.NORMAL.value: 0.70,
    OrderPriority.HIGH.value: 0.20,
    OrderPriority.URGENT.value: 0.10,
}


def generate_orders(
    config: GeneratorConfig,
    coils: pd.DataFrame,
    vehicles: pd.DataFrame,
) -> pd.DataFrame:
    """Siparişleri üretir ve her siparişe bobin + araç atar.

    Her sipariş benzersiz bir araçla eşleştirilir (araçlar tekrar kullanılmaz).
    Bobinler, aracın kapasitesini aşmayacak şekilde rastgele havuzdan seçilir;
    seçilen bobinin ``order_id``si yerinde güncellenerek çift yönlü bağ kurulur.
    Determinizm ``config.seed`` ile sağlanır.

    YAN ETKİ: ``coils`` DataFrame'inin ``order_id`` sütununu günceller.
    Dönüş: Order şemasına uygun DataFrame (``coil_ids`` liste sütunudur).
    """
    rng = np.random.default_rng(config.seed)

    # Sipariş sayısı araç sayısıyla sınırlıdır (her sipariş ayrı araç kullanır).
    n_orders = min(config.n_orders, len(vehicles))

    # Araçları karıştırıp ilk n_orders tanesini siparişlere ayır (tekrarsız).
    vehicle_order = rng.permutation(len(vehicles))[:n_orders]

    # Bobin havuzu: konumsal indekslerin karıştırılmış listesi. Atanan bobin
    # havuzdan çıkarılır, böylece bir bobin birden çok siparişe düşmez.
    pool = list(rng.permutation(len(coils)))

    # Önceden çekilen rastgele diziler (rng çağrı sırasını sabit tutmak için).
    priorities = rng.choice(
        list(PRIORITY_DISTRIBUTION.keys()),
        size=n_orders,
        p=list(PRIORITY_DISTRIBUTION.values()),
    )
    lead_days = rng.integers(1, 11, size=n_orders)  # deadline tamponu: 1..10 gün

    # Hızlı erişim için numpy/Series görünümleri.
    coil_weight = coils["weight_ton"].to_numpy()
    coil_prod = coils["production_time"].to_numpy()  # datetime64 dizisi
    coil_id_arr = coils["coil_id"].to_numpy()
    veh_capacity = vehicles["max_weight_capacity_ton"].to_numpy()
    veh_planned = vehicles["planned_arrival"].to_numpy()
    veh_id_arr = vehicles["vehicle_id"].to_numpy()

    # coils['order_id'] sütununu liste olarak alıp güncelleyeceğiz (yan etki).
    order_id_col: list[str | None] = list(coils["order_id"])

    rows: list[dict] = []
    for k in range(n_orders):
        vidx = int(vehicle_order[k])
        capacity = float(veh_capacity[vidx])
        desired_k = int(rng.integers(1, MAX_COILS_PER_ORDER + 1))

        # Havuzdan kapasiteye sığan bobinleri açgözlü topla. Sığmayan bobin
        # havuzda kalır (sonraki siparişlere). En hafif bobin (>=10 ton) en küçük
        # kapasiteye (TIR ~22 ton) sığdığından her sipariş en az 1 bobin alır.
        chosen: list[int] = []
        chosen_weight = 0.0
        i = 0
        while i < len(pool) and len(chosen) < desired_k:
            ci = pool[i]
            w = float(coil_weight[ci])
            if chosen_weight + w <= capacity:
                chosen.append(ci)
                chosen_weight += w
                pool.pop(i)  # bobini havuzdan çıkar; i aynı kalır (liste kayar)
            else:
                i += 1

        if not chosen:
            # Havuz tükendi — daha fazla sipariş üretilemez.
            break

        order_id = f"ORD-{k + 1:06d}"
        # Çift yönlü bağ: seçilen her bobinin order_id'si bu siparişe işaret eder.
        for ci in chosen:
            order_id_col[ci] = order_id

        # deadline hem en geç üretim zamanından hem aracın varışından sonradır.
        latest_prod = max(coil_prod[ci] for ci in chosen)
        planned = veh_planned[vidx]
        base_time = max(pd.Timestamp(latest_prod), pd.Timestamp(planned))
        deadline = base_time + pd.Timedelta(days=int(lead_days[k]))

        rows.append(
            {
                "order_id": order_id,
                "vehicle_id": str(veh_id_arr[vidx]),
                "coil_ids": [str(coil_id_arr[ci]) for ci in chosen],
                "deadline": deadline,
                "priority": priorities[k],
                "status": OrderStatus.OPEN.value,
            }
        )

    # Yan etkiyi uygula: güncellenmiş order_id sütununu coils'e geri yaz.
    coils["order_id"] = order_id_col

    return pd.DataFrame(rows)
