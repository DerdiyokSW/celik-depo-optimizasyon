"""PPO ödül fonksiyonu — ayrı dosyada, ayrı test edilebilir (docs/07 §7).

Üç bileşen: (1) yoğun ama küçük yönlendirme ödülü (erken öğrenmeyi hızlandırır),
(2) gerçekleşen sevkiyat sinyali (asıl hedef: rehandling + mesafe), (3) bölüm sonu
terminal ödülü (baseline ile kıyas). Ağırlıklar, adım başına ödülü kabaca [-2, 2]
aralığında tutacak şekilde seçilmiştir.

ÖNEMLİ: Geçersiz hamle cezası YOKTUR — geçersiz eylemler action masking ile zaten
elenir; ödül fonksiyonu fizik ihlalini hiç görmez.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain import SlotCoord, SteelCoil
from src.policies.scoring import planned_urgency, score_slot
from src.simulation.metrics import crane_distance

if TYPE_CHECKING:
    from src.simulation.simulator import WarehouseSimulator

# --- Ödül ağırlıkları (v3: DOĞRUDAN + normalize vinç maliyeti) ---
# Evrim: v1 yalnızca rehandling'i hedefledi → DEJENERE "hiç istifleme" (kapasite israfı).
# v2 toplam vinç mesafesini hedefledi ama sinyal gecikmiş/gürültülüydü (dispatch mesafesi
# yanlış adımlara dağılıyor) → yavaş öğrenme, kapı-yakınlığını öğrenemedi.
# v3: yerleştirme anında DOĞRUDAN maliyet — "girişten slota + slottan kapıya" — anında
# ödüllenir (kredi-atama kolay) ve normalize edilir (kararlı/hızlı öğrenme). Bu, ajanı
# DOĞRU ZONE İÇİNDE (affinity zaten zorlu) kapıya yakın + düşük kata + gömmeden istif
# yapmaya iter — gerçek operasyon becerisi.
W_GUIDE: float = 0.3     # yönlendirme: score_slot (kapı-aciliyet + istif disiplini)
W_COST: float = 1.0      # yerleştirmenin DOĞRUDAN vinç maliyeti (normalize)
W_REH: float = 1.0       # rehandling cezası (istif hatası: sevkiyatı yakın bobini gömme)
W_TERM: float = 0.3      # terminal: baseline'a göre toplam vinç mesafesi (holistik küçük sinyal)
COST_NORM_M: float = 80.0  # tipik yerleştirme+kapı maliyeti normalizasyonu (m)

_ENTRY_SLOT = SlotCoord(0, 0, 0)  # üretim bandı çıkışı (vincin bobini aldığı yer)


def placement_cost(slot: SlotCoord) -> float:
    """Bir yerleştirmenin DOĞRUDAN vinç maliyeti (m): girişten slota + slottan kapıya.

    İki gerçek hamle: (1) üretim çıkışından (0,0,0) seçilen slota, (2) slottan o bobinin
    sevkiyat kapısına (zone'un bay 0'ı). Toplamı, düşük bay (kapıya yakın) + düşük katı
    ANINDA ödüllendirir — gecikmeden, doğrudan. Heuristic'in score_slot'la yaptığı şeyin
    vinç-mesafesi karşılığı.
    """
    door = SlotCoord(slot.zone, 0, 0)
    return crane_distance(_ENTRY_SLOT, slot) + crane_distance(slot, door)


def guidance_reward(coil: SteelCoil, slot: SlotCoord, sim: "WarehouseSimulator") -> float:
    """``score_slot`` ile AYNI hedefe yönlendiren ödül (kapı-aciliyet + istif disiplini).

    Doğrudan maliyet hedefiyle uyumludur (acil bobini kapıya yakın koymak maliyeti düşürür).
    W_GUIDE erken evrede güçlü yön verir.
    """
    urgency = planned_urgency(coil, sim)
    raw = score_slot(coil, slot, urgency, sim, sim.layout)  # ~[-1, 2]
    return W_GUIDE * raw


def realized_reward(rehandling_delta: int, slot: SlotCoord) -> float:
    """O yerleştirmenin DOĞRUDAN maliyeti (normalize) + rehandling cezası.

    Yerleştirme anında bilinen (girişten slota + slottan kapıya) maliyeti hemen cezalandırır;
    bu, ajanın kapı-yakınlığı + düşük kat öğrenmesini hızlandırır. Rehandling (gecikmeli,
    istif hatası) ayrıca cezalanır.
    """
    return -W_COST * (placement_cost(slot) / COST_NORM_M) - W_REH * rehandling_delta


def terminal_reward(baseline_distance: float, agent_distance: float) -> float:
    """Bölüm sonu: ajanın TOPLAM VİNÇ MESAFESİNİ baseline (rastgele) ile kıyaslar (holistik).

    Pozitif = baseline'dan az toplam vinç işi = iyi. Küçük ağırlık (W_TERM); asıl sinyal
    yoğun doğrudan-maliyettir.
    """
    return W_TERM * (baseline_distance - agent_distance) / max(baseline_distance, 1.0)
