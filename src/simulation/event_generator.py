"""EventGenerator — simülasyon zaman ekseninde dinamik olay akışı üretir.

Olay zamanlaması Poisson süreciyle modellenir: olaylar arası süre üstel
dağılımlıdır (ortalama = 1/hız). Olay tipi ise verilen olasılık karışımından
seçilir. Bu, projenin stokastik/dinamik doğasının (gecikme, iptal, yeni sipariş,
zirve) kaynağıdır. Üreteç saf bir zaman/tip üreticisidir — hangi somut siparişe/
araca karşılık geldiğini simülatör bağlar (payload'ı simülatör doldurur).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.domain import Event, EventType

# Varsayılan olay tipi karışımı (docs/03 §8, rapordaki dağılımla uyumlu).
DEFAULT_TYPE_MIX: dict[EventType, float] = {
    EventType.NEW_ORDER: 0.55,
    EventType.VEHICLE_DELAY: 0.25,
    EventType.CANCEL_ORDER: 0.10,
    EventType.PRIORITY_CHANGE: 0.10,
}


class EventGenerator:
    """Poisson süreçli dinamik olay üreteci.

    Parametreler:
        rate_per_hour: Saatte beklenen olay sayısı (Poisson yoğunluğu).
        type_mix: Olay tipi -> olasılık karışımı (toplamı ~1). None ise varsayılan.
        seed: Determinizm tohumu (simülatörden ayrı bir akış).
    """

    def __init__(
        self,
        rate_per_hour: float = 12.0,
        type_mix: dict[EventType, float] | None = None,
        seed: int = 7,
    ) -> None:
        if rate_per_hour <= 0:
            raise ValueError("rate_per_hour pozitif olmalı.")
        self.rate_per_hour = rate_per_hour
        mix = type_mix if type_mix is not None else DEFAULT_TYPE_MIX
        # Tip ve olasılıkları sabit sıralı dizilere ayır; choice'u indeks üzerinden
        # yaparak str-enum'ların numpy'da metne dönüşmesi sorununu tamamen atlarız.
        self._types: list[EventType] = list(mix.keys())
        self._probs: list[float] = list(mix.values())
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def stream(self, horizon_hours: float) -> Iterator[Event]:
        """Ufuk boyunca zaman-sıralı olay akışı üretir (generator).

        Olaylar arası süre Exponential(1/rate) ile çekilir; birikimli zaman ufku
        aşınca durulur. Üretilen olayların ``payload``'ı boştur (simülatör doldurur).
        """
        rng = np.random.default_rng(self.seed)  # her stream çağrısı aynı diziyi verir
        t = 0.0
        while True:
            # Üstel dağılımlı bekleme süresi (saat). Ortalama = 1/rate.
            t += float(rng.exponential(1.0 / self.rate_per_hour))
            if t >= horizon_hours:
                return
            event_type = self._types[int(rng.choice(len(self._types), p=self._probs))]
            yield Event(timestamp=t, event_type=event_type, payload={})

    def trigger_peak(self, burst_factor: float = 5.0) -> Event:
        """Kriz/zirve senaryosu olayı üretir (Knapp entegrasyonu).

        Olay hızını ve sipariş aciliyetini geçici olarak artıracak bir PEAK_LOAD
        olayıdır. Zaman damgası nominaldir (0.0); simülatör olayı enjekte ederken
        anlık simülasyon saatini atar.
        """
        return Event(
            timestamp=0.0,
            event_type=EventType.PEAK_LOAD,
            payload={"burst_factor": burst_factor},
        )
