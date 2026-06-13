"""Veri üretiminin tüm ayarlanabilir parametreleri ve ortak zaman çıpası.

``GeneratorConfig`` tek bir yerden yönetim sağlar; tekrarlanabilirlik ``seed``
ile garanti edilir. ``HISTORY_START`` ise bobin üretim zamanları ile araç varış
zamanlarının aynı 12 aylık pencereye oturmasını sağlayan paylaşılan referans
noktasıdır — determinizm için asla ``datetime.now()`` kullanılmaz, sabit bir
çıpa kullanılır.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

# Geçmiş veri penceresinin sabit başlangıcı. Hem bobin üretim zamanları hem araç
# planlanan varışları bu çıpadan itibaren n_months aylık aralığa yayılır. Sabit
# olması, aynı seed'in her zaman aynı tarihleri üretmesini (determinizm) sağlar.
HISTORY_START: datetime = datetime(2025, 1, 1)

# Bir ayı kaç dakika sayacağımızın sabiti (basitleştirilmiş 30 günlük ay).
MINUTES_PER_MONTH: int = 30 * 24 * 60


@dataclass
class GeneratorConfig:
    """Veri üretiminin tüm ayarlanabilir parametrelerini tutar.

    Tek bir yerden yönetim sağlar; deney tekrarlanabilirliği ``seed`` ile garanti
    edilir (aynı seed -> aynı veri seti).

    Alanlar:
        n_coils: Üretilecek toplam bobin sayısı.
        n_orders: Üretilecek hedef sipariş sayısı (araç sayısıyla sınırlanabilir).
        n_months: Araç geçmişi ve bobin üretiminin kaç aya yayılacağı.
        n_vehicles: Üretilecek geçmiş araç kaydı sayısı (ML eğitimi için).
        seed: Tüm rastgeleliğin tek tohumu.
    """

    n_coils: int = 5000
    n_orders: int = 1200
    n_months: int = 12
    n_vehicles: int = 3600
    seed: int = 42

    def window_minutes(self) -> int:
        """Geçmiş veri penceresinin toplam uzunluğunu dakika cinsinden döndürür."""
        return self.n_months * MINUTES_PER_MONTH
