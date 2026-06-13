"""Tohumlanmış senaryo üretimi — adil karşılaştırmanın temeli.

Karşılaştırmanın geçerli olması için dört politikanın da BİREBİR aynı senaryoları
görmesi şarttır. Bir ``ScenarioSpec`` simülatörü ve event generator'ı deterministik
kurmak için gereken tüm tohumları/parametreleri taşır; ``make_scenarios`` aynı
``base_seed`` ile her zaman aynı kümeyi üretir.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    """Tek bir tekrarlanabilir değerlendirme senaryosu.

    Alanlar:
        event_seed: EventGenerator tohumu (dinamik olay akışı).
        sim_seed: Simülatör tohumu (teslim süresi/gecikme/iptal seçimleri).
        event_rate_per_hour: Olay yoğunluğu (dayanıklılık analizinde değişir).
        horizon_hours: Bölüm ufku.
    """

    event_seed: int
    sim_seed: int
    event_rate_per_hour: float = 12.0
    horizon_hours: float = 24.0


def make_scenarios(
    n: int,
    base_seed: int = 1000,
    event_rate_per_hour: float = 12.0,
    horizon_hours: float = 24.0,
) -> list[ScenarioSpec]:
    """N adet tekrarlanabilir senaryo üretir (aynı base_seed -> aynı küme).

    Her senaryo benzersiz ama deterministik tohumlar alır; tüm politikalar bu aynı
    listeyi kullanır, böylece karşılaştırma eşleştirilmiş (paired) olur.
    """
    return [
        ScenarioSpec(
            event_seed=base_seed + i,
            sim_seed=base_seed + i,
            event_rate_per_hour=event_rate_per_hour,
            horizon_hours=horizon_hours,
        )
        for i in range(n)
    ]
