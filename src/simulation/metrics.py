"""Metrik hesaplama: vinç mesafesi (Manhattan), hareket süresi ve koşu metrikleri.

Vinç mesafesi modeli docs/01 §6 J4n1k entegrasyonundan gelir: depo içi hareket
maliyeti Manhattan mesafesiyle temsil edilir. Tüm ölçek sabitleri burada
adlandırılmış ve yorumlanmış olarak tutulur (koda gömülü sihirli sayı yok).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain import SlotCoord

# --- Mesafe ölçek sabitleri (metre). Zone ve bay iki yatay eksen, layer dikey. ---
ZONE_SPACING_M: float = 15.0   # zone ekseni boyunca iki konum arası mesafe
BAY_SPACING_M: float = 3.0     # bay ekseni boyunca iki konum arası mesafe
LAYER_HEIGHT_M: float = 1.5    # iki kat arası dikey mesafe

# Yatay mesafe modeli — ileride canlıda sorun çıkarsa değiştirilebilsin diye anahtarlı.
#   "chebyshev": tavan (köprü) vinci modeli — köprü ve troley AYNI ANDA hareket eder,
#                yatay maliyet = max(zone ekseni, bay ekseni). Fiziksel olarak gerçekçi.
#   "manhattan": iki yatay eksen toplanır (zone + bay). Literatürde yaygın, basit.
HORIZONTAL_DISTANCE_MODE: str = "chebyshev"

# --- Yükleme/hareket süre modeli (docs/03 §7) ---
CRANE_SETUP_MIN: float = 1.5       # her hamlenin sabit hazırlık süresi (dakika)
CRANE_TIME_PER_M: float = 0.05     # mesafeyle orantılı süre (dakika/metre)


def crane_distance(a: SlotCoord, b: SlotCoord) -> float:
    """İki konum arası vinç hareket mesafesi (metre).

    Yatay bileşen ``HORIZONTAL_DISTANCE_MODE``a göre hesaplanır (tavan vinci için
    Chebyshev = eşzamanlı köprü+troley, ya da Manhattan = eksenler toplamı); buna
    dikey kaldırma (layer) eklenir. Sevkiyat ve yerleştirme hamlelerinin maliyetidir.
    """
    zone_dist = ZONE_SPACING_M * abs(a.zone - b.zone)
    bay_dist = BAY_SPACING_M * abs(a.bay - b.bay)
    if HORIZONTAL_DISTANCE_MODE == "chebyshev":
        horizontal = max(zone_dist, bay_dist)  # köprü+troley aynı anda hareket eder
    else:
        horizontal = zone_dist + bay_dist
    vertical = LAYER_HEIGHT_M * abs(a.layer - b.layer)
    return horizontal + vertical


def crane_move_time(distance_m: float) -> float:
    """Bir vinç hamlesinin süresini (dakika) döndürür: sabit hazırlık + mesafe payı."""
    return CRANE_SETUP_MIN + distance_m * CRANE_TIME_PER_M


@dataclass
class SimulationMetrics:
    """Bir simülasyon koşusunun tüm performans metrikleri.

    Politikaların karşılaştırılacağı (Aşama 7) çekirdek çıktısıdır. Alanlar:
        rehandling_count: Toplam rehandling (engelleyici bobin oynatma) sayısı — ana maliyet.
        total_crane_distance_m: Tüm vinç hareketlerinin toplam mesafesi (m).
        total_loading_time_min: Sevkiyat (yükleme) hamlelerinin toplam süresi (dk).
        final_fill_ratio: Koşu sonunda depo doluluk oranı (0..1).
        n_placements: Yapılan yerleştirme sayısı.
        n_dispatches: Tamamlanan sevkiyat (dispatch) sayısı.
        decision_times_ms: Her yerleştirme kararının hesaplama süresi (ms) — hız analizi.
    """

    rehandling_count: int = 0
    total_crane_distance_m: float = 0.0
    total_loading_time_min: float = 0.0
    final_fill_ratio: float = 0.0
    n_placements: int = 0
    n_dispatches: int = 0
    decision_times_ms: list[float] = field(default_factory=list)
