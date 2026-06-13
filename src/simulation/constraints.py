"""Fizik/istif kısıt kuralları — yan etkisiz saf fonksiyonlar.

Bir bobinin bir konuma konulabilirliğini docs/03 §5'teki 5 kurala (ve konum
sınırlarına) göre denetler. Saf ve test edilmesi kolaydır; ``WarehouseState`` bu
fonksiyonları ``valid_slots`` içinde kullanır, dolayısıyla import yönü tek
taraflıdır (state -> constraints). Döngüsel import'tan kaçınmak için buradaki
``WarehouseState`` referansı yalnızca tip denetimi içindir.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain import SlotCoord, SteelCoil

if TYPE_CHECKING:
    from .warehouse_state import WarehouseState


def can_place(state: "WarehouseState", coil: SteelCoil, slot: SlotCoord) -> bool:
    """Tüm kısıtlar sağlanıyorsa True. Detaylı sebep için ``placement_violations``."""
    return not placement_violations(state, coil, slot)


def placement_violations(
    state: "WarehouseState", coil: SteelCoil, slot: SlotCoord
) -> list[str]:
    """İhlal edilen kuralların Türkçe açıklamalarını liste olarak döndürür (boşsa geçerli).

    Kurallar (docs/01 ile birebir):
      0. Konum depo sınırları içinde olmalı (geçersizse erken döner).
      1. Boşluk: hedef konum boş olmalı.
      2. Maks kat: layer < coil.max_stack_layer.
      3. Süreklilik: layer > 0 ise alt kat dolu olmalı.
      4. Ağırlık: layer > 0 ise alttaki bobin bu bobinden ağır olmalı.
      5. Zone kapasitesi: konum bobini alınca zone toplam tonajı limiti aşmamalı.
    """
    violations: list[str] = []
    layout = state.layout

    # 0. Sınır kontrolü — geçersiz konumda diğer kuralları denetlemek anlamsız.
    if not layout.is_valid_coord(slot):
        violations.append("konum depo sınırları dışında")
        return violations

    # 1. Boşluk.
    if not state.is_empty(slot):
        violations.append("hedef konum dolu")

    # 2. Maksimum kat (bobin tipinin izin verdiği üst sınır).
    if slot.layer >= coil.max_stack_layer:
        violations.append("maksimum istif katı aşıldı")

    # 3 & 4. Üst katlar için süreklilik ve ağırlık.
    if slot.layer > 0:
        below_slot = SlotCoord(slot.zone, slot.bay, slot.layer - 1)
        below_coil = state.coil_at(below_slot)
        if below_coil is None:
            violations.append("istif sürekli değil (alt kat boş)")
        elif not (coil.weight_ton < below_coil.weight_ton):
            # Üst kat kesinlikle alttan hafif olmalı (ağır alta).
            violations.append("ağırlık kuralı ihlali (üst kat alttan hafif olmalı)")

    # 5. Zone kapasitesi.
    if state.zone_weight(slot.zone) + coil.weight_ton > layout.zone_max_weight_ton[slot.zone]:
        violations.append("zone kapasite limiti aşıldı")

    return violations
