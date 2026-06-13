"""Klasik ve ML-destekli sezgisellerin paylaştığı saf skorlama fonksiyonları.

İki sezgisel BİREBİR aynı skorlamayı kullanır; aralarındaki tek fark aciliyetin
nasıl hesaplandığıdır (ML, gecikme tahminini ekler). Bu, ML'in net katkısını
izole etmek için bilinçli bir tasarımdır. Tüm ağırlık sabitleri burada
adlandırılmış ve yorumlanmış tutulur (sihirli sayı yok).

Felsefe (docs/00 §6): yerleştirme "zaman-öncelikli ve katmanlı"dır — sevkiyatı
yakın (acil) bobinler erişilebilir konuma, uzak olanlar derine. Affinity yalnızca
ikincil bir bileşendir.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain import SlotCoord, SteelCoil, WarehouseLayout

if TYPE_CHECKING:
    from src.simulation.simulator import WarehouseSimulator

# Aciliyetin doygunluğa ulaştığı ufuk (saat). Sevkiyatına bu süreden az kalan
# bobinler kademeli olarak "acil" sayılır; daha uzak olanların aciliyeti 0'dır.
# Simülatörün planlanan teslim süresi aralığıyla (~1-12 saat + gecikme) uyumludur.
URGENCY_HORIZON_HOURS: float = 12.0

# Erişilebilirlik bileşen ağırlıkları (slot_accessibility; reward yönlendirmesi kullanır).
W_ACC_LAYER: float = 0.6   # üst kat = üstünde yük yok = erişilebilir
W_ACC_EXIT: float = 0.4    # zone çıkışına (bay 0) yakın = erişilebilir

# Toplam skor ağırlıkları (score_slot).
W_DOOR: float = 1.0        # kapı yakınlığı (aciliyetle ölçekli): acil bobin kapıya
W_STACK: float = 0.7       # istif disiplini: zemin tercih, istiflenecekse acil üste
W_AFFINITY: float = 0.3    # lojistik hattı zone eşleşmesi (ikincil)


def coil_urgency(hours_to_dispatch: float) -> float:
    """Sevkiyata kalan süreden 0..1 aciliyet skoru üretir (yakınsa yüksek).

    urgency = clip(1 - hours_to_dispatch / URGENCY_HORIZON, 0, 1).
    """
    return float(min(1.0, max(0.0, 1.0 - hours_to_dispatch / URGENCY_HORIZON_HOURS)))


def planned_urgency(coil: SteelCoil, sim: "WarehouseSimulator") -> float:
    """Aciliyeti, siparişin PLANLANAN sevkiyatına KALAN süreden hesaplar (gecikmesiz).

    hours_to_dispatch = planlanan_dispatch - şu anki sim saati. Yakında sevk edilecek
    bobin daha aciledir → erişilebilir konuma (kapıya yakın/üst kat) yerleştirilmeli.
    Sipariş etkin değilse (planlanan dispatch yok) aciliyet 0'dır. HeuristicPolicy ve
    PPO ödülünün yönlendirme terimi bu ortak hesabı paylaşır.
    """
    planned = sim.planned_dispatch_time(sim.order_of(coil))
    if planned is None:
        return 0.0
    hours_to_dispatch = planned - sim.clock
    return coil_urgency(max(0.0, hours_to_dispatch))


def slot_accessibility(slot: SlotCoord, layout: WarehouseLayout) -> float:
    """Bir konumdan bobin almanın kolaylığını 0..1 ile ölçer.

    Üst kat (üstünde yük yok) ve zone çıkışına (bay 0) yakınlık erişilebilirliği
    artırır. accessibility = W_ACC_LAYER * kat_skoru + W_ACC_EXIT * çıkış_yakınlığı.
    """
    layer_score = slot.layer / (layout.n_layers - 1) if layout.n_layers > 1 else 1.0
    exit_score = 1.0 - slot.bay / (layout.n_bays - 1) if layout.n_bays > 1 else 1.0
    return W_ACC_LAYER * layer_score + W_ACC_EXIT * exit_score


def placement_fit(urgency: float, accessibility: float) -> float:
    """Aciliyet ile erişilebilirliğin ne kadar ÖRTÜŞTÜĞÜ (0..1).

    Acil bobin erişilebilir konuma, acil olmayan derine konmalı. fit = 1 - |fark|.
    """
    return 1.0 - abs(urgency - accessibility)


def affinity_bonus(coil: SteelCoil, slot: SlotCoord, sim: "WarehouseSimulator") -> float:
    """Bobin, kendi lojistik hattına hizmet eden zone'a konursa +1, değilse 0.

    J4n1k affinity fikrinin uyarlaması: aynı araca/hatta gidecek bobinler aynı
    zone'da toplanır (sevkiyatta birlikte alınır -> rehandling azalır).
    """
    vehicle = sim.vehicle_of(sim.order_of(coil))
    if vehicle is None:
        return 0.0
    return 1.0 if sim.layout.zone_logistics.get(slot.zone) == vehicle.target_logistics_line else 0.0


def score_slot(
    coil: SteelCoil,
    slot: SlotCoord,
    urgency: float,
    sim: "WarehouseSimulator",
    layout: WarehouseLayout,
    urgency_fn=planned_urgency,
) -> float:
    """Bir aday konumun toplam skoru (yüksek = daha iyi).

    Üç bileşen:
      1) Kapı yakınlığı (aciliyetle ölçekli): acil bobin yükleme kapısına (bay 0)
         yakın konmalı; acil olmayanın kapıya yakınlığı önemsenmez (uzağa yayılır).
      2) İstif disiplini: zemin (gömme yaratmaz) tercih edilir; üst kata yalnızca
         alttakinden daha acil/eşit bir bobin konursa iyidir (acil üste = ilk sevk).
      3) Affinity: bobin kendi lojistik hattının zone'una konarsa bonus.

    ``urgency_fn``: alttaki bobinin aciliyetini, yerleştirilen bobinle AYNI ölçüyle
    hesaplamak için kullanılır (Heuristic planlanan, ML gecikme-düzeltilmiş kullanır;
    karşılaştırmanın tutarlı olması şart).
    """
    # 1) Kapı–aciliyet eşleşmesi: acil bobin kapıya (bay 0) yakın, acil olmayan
    #    uzağa (yüksek bay) konmalı — kapı yakınını acil sevkiyatlar için boş tutar.
    door_proximity = 1.0 - slot.bay / (layout.n_bays - 1) if layout.n_bays > 1 else 1.0
    door_term = 1.0 - abs(urgency - door_proximity)

    # 2) İstif disiplini (alt bobinin aciliyeti, yerleştirilenle aynı ölçüyle).
    if slot.layer == 0:
        stack_term = 1.0  # zemin: kimseyi gömmez
    else:
        below = sim.state.coil_at(SlotCoord(slot.zone, slot.bay, slot.layer - 1))
        below_urgency = urgency_fn(below, sim) if below is not None else 0.0
        # Acil/eşit bobini üste koymak iyi; daha acil bir bobini gömmek kötü.
        stack_term = 1.0 if urgency >= below_urgency else -1.0

    affinity = affinity_bonus(coil, slot, sim)
    return W_DOOR * door_term + W_STACK * stack_term + W_AFFINITY * affinity
