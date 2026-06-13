"""HeuristicPolicy — klasik, açıklanabilir, kuralı net sezgisel politika.

Aciliyet–erişilebilirlik örtüşmesi + affinity ile en yüksek skorlu geçerli konumu
seçer. Aciliyet, aracın PLANLANAN varış zamanına dayanır (gecikme tahmini
KULLANILMAZ). Literatürdeki öncelik tabanlı yerleştirmenin "zaman-öncelikli
katmanlı yerleştirme" olarak yeniden formüle edilmiş hâlidir.

SWAP MEKANİZMASI (Paket 3): Yeni gelen ACİL bir bobin için en iyi (kapıya yakın,
istif tepesinde) konum, daha az acil bir bobinle dolu ise — o bobini uygun bir
alternatife taşıyıp (relocate, 1 vinç hamlesi) yeni bobini prime'a koymak,
yeni bobini suboptimal yere koymaktan daha az toplam vinç işi gerektiriyorsa
swap yapılır (B1 denklemi). PPO bu mekanizmayı KULLANMAZ (aksiyon uzayı slot
seçimi; swap ayrı bir aksiyon tipi olur).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain import SlotCoord, SteelCoil
from src.simulation.constraints import can_place
from src.simulation.metrics import crane_distance

from .base import PlacementPolicy
from .scoring import planned_urgency, score_slot

if TYPE_CHECKING:
    from src.simulation.simulator import WarehouseSimulator

# Swap mekanizması parametreleri.
SWAP_DOOR_BAYS: int = 6      # yalnızca ilk N bay (kapıya yakın) prime aday olabilir
SWAP_MIN_URGENCY_GAP: float = 0.15  # yeni - eski aciliyet farkı en az bu olmalı
SWAP_MIN_NEW_URGENCY: float = 0.3   # acil olmayan bobinler için swap denenmez

# B3 yeniden konumlandırma: bobini taşımak için skor kazanımı bu eşiği aşmalı
# (küçük kazanımlar için gereksiz vinç işi yapma). Yalnızca acil bobinlerde denenir.
REPOSITION_MIN_GAIN: float = 0.12
REPOSITION_MIN_URGENCY: float = 0.4

_ENTRY_SLOT = SlotCoord(0, 0, 0)


def _zone_exit(zone: int) -> SlotCoord:
    """Bir zone'un sevkiyat çıkış (rıhtım) noktası — B1 denkleminde kullanılır."""
    return SlotCoord(zone, 0, 0)


class HeuristicPolicy(PlacementPolicy):
    """Aciliyet + erişilebilirlik + affinity skorlamasıyla yerleştiren sezgisel.

    Swap mekanizması: ``decide`` önce en iyi boş konumu hesaplar, sonra daha iyi
    bir prime konumda daha az acil bir bobinle dolu olup olmadığını arar; B1
    denklemiyle swap'ın vinç işi açısından kazançlı olduğu durumda relocate çağırıp
    prime'ı boşaltır.
    """

    def decide(self, coil: SteelCoil, sim: "WarehouseSimulator") -> SlotCoord:
        """En yüksek skorlu konumu döndürür; uygun ise swap yapar."""
        valid = sim.valid_actions()
        urgency = self._compute_urgency(coil, sim)
        # En yüksek skorlu boş slot (swap'sız yerleştirme hedefi).
        best_empty = max(
            valid,
            key=lambda slot: score_slot(coil, slot, urgency, sim, sim.layout, self._compute_urgency),
        )
        # Swap fırsatı var mı?
        swap = self._consider_swap(coil, urgency, best_empty, sim)
        if swap is not None:
            prime_slot, old_coil, alt_slot = swap
            # Kararın gerekçesini (ve B1 denkleminin iki tarafını) taşınan bobine kaydet;
            # görselleştirme bunu elmas hover'ında "niye taşındı" olarak gösterir (B1).
            swap_cost, alt_cost = _swap_costs(prime_slot, alt_slot, best_empty)
            old_coil.swap_reason = {
                "trigger_coil": coil.coil_id,            # bu acil bobin için yer açıldı
                "trigger_urgency": round(urgency, 2),
                "moved_from": (prime_slot.zone, prime_slot.bay, prime_slot.layer),
                "moved_to": (alt_slot.zone, alt_slot.bay, alt_slot.layer),
                "swap_cost_m": round(swap_cost, 1),       # B1 sol taraf (swap maliyeti)
                "alt_cost_m": round(alt_cost, 1),         # B1 sağ taraf (swap olmasaydı)
            }
            sim.relocate(old_coil, alt_slot)
            return prime_slot
        return best_empty

    def _compute_urgency(self, coil: SteelCoil, sim: "WarehouseSimulator") -> float:
        """Aciliyeti aracın PLANLANAN varışından hesaplar (ML kullanmaz).

        Ortak ``scoring.planned_urgency`` hesabını kullanır; MLHeuristicPolicy bu
        metodu override ederek gecikme tahminini ekler (tek kod farkı).
        """
        return planned_urgency(coil, sim)

    # ------------------------------------------------------------------ swap
    def _consider_swap(
        self,
        new_coil: SteelCoil,
        new_urgency: float,
        best_empty: SlotCoord,
        sim: "WarehouseSimulator",
    ) -> tuple[SlotCoord, SteelCoil, SlotCoord] | None:
        """Kapıya yakın, daha az acil bir bobinle dolu prime slot ara; varsa swap üçlüsünü döndür."""
        if new_urgency < SWAP_MIN_NEW_URGENCY:
            return None  # acil olmayan bobin için swap düşünme (vinç işi boşa)

        layout = sim.layout
        vehicle = sim.vehicle_of(sim.order_of(new_coil))
        target_line = vehicle.target_logistics_line if vehicle is not None else None

        no_swap_score = score_slot(
            new_coil, best_empty, new_urgency, sim, layout, self._compute_urgency
        )

        best: tuple[SlotCoord, SteelCoil, SlotCoord, float] | None = None
        door_bays = min(SWAP_DOOR_BAYS, layout.n_bays)
        for zone in range(layout.n_zones):
            # Yalnızca affinity zone'larına bak (mevcutsa).
            if target_line is not None and layout.zone_logistics.get(zone) != target_line:
                continue
            for bay in range(door_bays):
                for layer in range(layout.n_layers):
                    prime_slot = SlotCoord(zone, bay, layer)
                    occupant = sim.state.coil_at(prime_slot)
                    if occupant is None:
                        continue
                    old_urgency = self._compute_urgency(occupant, sim)
                    # Eski bobin yeterince daha az acil olmalı.
                    if new_urgency - old_urgency < SWAP_MIN_URGENCY_GAP:
                        continue
                    # Prime'da skor, best_empty'den iyi olmalı (kazanım olmalı).
                    prime_score = score_slot(
                        new_coil, prime_slot, new_urgency, sim, layout, self._compute_urgency
                    )
                    if prime_score <= no_swap_score:
                        continue
                    # Eski bobini geçici çıkar; (1) en iyi alternatifini bul, (2) yeni acil
                    # bobinin boşalan prime'a FİİLEN konabildiğini doğrula (kritik: aksi hâlde
                    # apply_placement geçersiz konumla patlar).
                    alt_slot = self._evaluate_vacated(new_coil, occupant, prime_slot, sim)
                    if alt_slot is None or alt_slot == prime_slot:
                        continue
                    # B1: vinç işi açısından swap kazançlı mı?
                    if not _swap_worthwhile(prime_slot, alt_slot, best_empty):
                        continue
                    gain = prime_score - no_swap_score
                    if best is None or gain > best[3]:
                        best = (prime_slot, occupant, alt_slot, gain)

        if best is None:
            return None
        return best[0], best[1], best[2]

    def _evaluate_vacated(
        self,
        new_coil: SteelCoil,
        occupant: SteelCoil,
        prime_slot: SlotCoord,
        sim: "WarehouseSimulator",
    ) -> SlotCoord | None:
        """Occupant'ı geçici çıkarıp swap'ın FİZİKSEL GEÇERLİLİĞİNİ doğrular.

        Gerçek yürütme sırası: önce occupant prime'dan alternatife taşınır, SONRA yeni
        acil bobin prime'a konur. Bu metot o sırayı birebir simüle eder:
          1) Occupant'ı çıkar, prime dışı en iyi alternatif slotu (alt) bul.
          2) Occupant'ı alt'a koy; bu durumdayken yeni bobin prime'a ``can_place`` mı?
             (Süreklilik/ağırlık/zone kapasitesi occupant alt'tayken doğru hesaplanır.)
        Geçerliyse alt slotu, değilse None döndürür. Durumu her hâlükârda eski hâline alır.
        """
        original = occupant.location
        if original is None:
            return None
        sim.state.remove(occupant)
        try:
            valid = sim.state.valid_slots(occupant)
            # Occupant prime'a geri dönmesin (anlamsız); prime dışı en iyi alternatif.
            # Affinity: occupant da yalnızca kendi hattının zone'larına taşınabilir.
            allowed = sim.allowed_zones(occupant)
            candidates = [
                s for s in valid
                if s != prime_slot and (allowed is None or s.zone in allowed)
            ]
            if not candidates:
                return None
            old_urgency = self._compute_urgency(occupant, sim)
            alt = max(
                candidates,
                key=lambda s: score_slot(occupant, s, old_urgency, sim, sim.layout, self._compute_urgency),
            )
            # Occupant'ı alt'a koy ve yeni bobinin prime'a gerçekten sığdığını doğrula.
            sim.state.place(occupant, alt)
            try:
                if not can_place(sim.state, new_coil, prime_slot):
                    return None
            finally:
                sim.state.remove(occupant)
            return alt
        finally:
            sim.state.place(occupant, original)

    # ------------------------------------------------------- B3 yeniden konumlandırma
    def reposition_on_priority_change(
        self, coil: SteelCoil, sim: "WarehouseSimulator"
    ) -> SlotCoord | None:
        """Aciliyeti yükselen, erişilebilir (istif tepesinde) bir bobini daha iyi bir
        boş konuma taşır (B3). Kazanç ``REPOSITION_MIN_GAIN``'i aşarsa relocate eder.

        Yalnızca istif TEPESİNDEKİ bobinleri taşır — gömülü (üstünde bobin olan) bir
        bobini taşımak diğer bobinleri de oynatmayı gerektirir (kapsam dışı). Güncel
        zaman-tabanlı aciliyet düşükse hiç denemez. Taşınan bobine görselleştirme için
        bir gerekçe (``swap_reason``, kind='reposition') yazar. Dönüş: yeni konum / None.
        """
        loc = coil.location
        if loc is None:
            return None
        # Yalnızca erişilebilir (tepe) bobin: stack_height == layer + 1.
        if sim.state.stack_height(loc.zone, loc.bay) != loc.layer + 1:
            return None
        urgency = self._compute_urgency(coil, sim)
        if urgency < REPOSITION_MIN_URGENCY:
            return None  # yeterince acil değil; taşıma vinç işi boşa
        current_score = score_slot(coil, loc, urgency, sim, sim.layout, self._compute_urgency)
        # Bobini geçici çıkarıp en iyi boş alternatifi bul (kapasite doğru hesaplansın).
        sim.state.remove(coil)
        try:
            candidates = [s for s in sim.state.valid_slots(coil) if s != loc]
            if not candidates:
                return None
            best = max(
                candidates,
                key=lambda s: score_slot(coil, s, urgency, sim, sim.layout, self._compute_urgency),
            )
            best_score = score_slot(coil, best, urgency, sim, sim.layout, self._compute_urgency)
        finally:
            sim.state.place(coil, loc)
        if best_score - current_score <= REPOSITION_MIN_GAIN:
            return None  # kazanım eşiğin altında; yerinde bırak
        # Taşı ve gerekçeyi işaretle (görselleştirme: elmas + hover).
        sim.relocate(coil, best)
        coil.swap_reason = {
            "kind": "reposition",
            "trigger_urgency": round(urgency, 2),
            "moved_from": (loc.zone, loc.bay, loc.layer),
            "moved_to": (best.zone, best.bay, best.layer),
        }
        return best

    @property
    def name(self) -> str:
        return "Heuristic"


def _swap_costs(
    prime: SlotCoord, alt: SlotCoord, best_empty: SlotCoord
) -> tuple[float, float]:
    """B1 denkleminin iki tarafını (swap_cost, no_swap) metre cinsinden döndürür.

    swap_cost  = d(prime, alt) + d_in(prime) + d_out(alt)
        (eski bobini prime'dan alt'a taşı + yeni bobini girişten prime'a koy +
         eski bobini alt'tan sevkiyat çıkışına götür)
    no_swap    = d_in(best_empty) + d_out(best_empty)
        (yeni bobini doğrudan best_empty'ye koy + oradan sevkiyat çıkışına götür)
    ``d_out(prime)`` iki tarafta da ortak olduğundan denklemden sadeleşir.
    """
    swap_cost = (
        crane_distance(prime, alt)
        + crane_distance(_ENTRY_SLOT, prime)
        + crane_distance(alt, _zone_exit(alt.zone))
    )
    no_swap = (
        crane_distance(_ENTRY_SLOT, best_empty)
        + crane_distance(best_empty, _zone_exit(best_empty.zone))
    )
    return swap_cost, no_swap


def _swap_worthwhile(prime: SlotCoord, alt: SlotCoord, best_empty: SlotCoord) -> bool:
    """B1 denklemi: swap toplam vinç işini azaltıyor mu? (swap_cost < no_swap)"""
    swap_cost, no_swap = _swap_costs(prime, alt, best_empty)
    return swap_cost < no_swap
