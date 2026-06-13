"""Curriculum learning — zorluk kademelendirme (docs/07 §9).

PPO'nun yakınsamasını kolaylaştırmak için eğitim, olay yoğunluğu artan kademelerde
yapılır: kolaydan zora. Doğrudan zor senaryoda eğitime kıyasla daha kararlı bir
öğrenme eğrisi verir. train_ppo bu kademeleri sırayla dolaşır.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    """Tek bir zorluk kademesi (olay hızıyla tanımlanır)."""

    name: str
    event_rate_per_hour: float
    description: str


# Kademeler: olay hızı arttıkça kuyruk yoğunlaşır, kararlar zorlaşır.
STAGES: list[CurriculumStage] = [
    CurriculumStage("Kolay", 8.0, "Düşük olay hızı, seyrek kuyruk, peak yok"),
    CurriculumStage("Orta", 12.0, "Orta olay hızı, ara sıra yoğunluk"),
    CurriculumStage("Zor", 20.0, "Yüksek olay hızı, sık peak, yoğun kuyruk"),
]
