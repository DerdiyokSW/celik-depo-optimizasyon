"""Sevkiyat (retrieval) mantığı ve rehandling sayımı — metriklerin kalbi.

Bir aracın varışında siparişinin bobinleri depodan alınır. Bir hedef bobinin
üstünde duran ve KENDİSİ bu sevkiyatın hedefi OLMAYAN bobinler, erişimi açmak için
geçici olarak alınıp kenara konmak (ve sonra geri istiflenmek) zorundadır; işte bu
zorunlu hamle **rehandling**tir. Hedefin üstündeki bobin de aynı sevkiyatın
hedefiyse, onu almak rehandling değil üretken hamledir (docs/03 §6).

Modelleme: engelleyiciler aynı sütuna geri istiflenir ("kazı ve yerine istifle").
Bu, çelik bobin/konteyner istif literatüründe standart bir varsayımdır ve dış boş
alan gerektirmediğinden depo dolu olsa bile sevkiyat her zaman uygulanabilirdir
(simülatör çökmez). Rehandling sayısı, taşınmak zorunda kalan engelleyici sayısıdır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain import Order, SlotCoord

from .metrics import crane_distance, crane_move_time
from .warehouse_state import WarehouseState


@dataclass
class DispatchResult:
    """Tek bir siparişin sevkiyatının sonucu."""

    rehandling_count: int = 0
    distance_m: float = 0.0
    loading_time_min: float = 0.0
    n_retrieved: int = 0
    # Hata ayıklama/görselleştirme için hamle kaydı: (tip, coil_id, kaynak, hedef).
    moves: list[tuple] = field(default_factory=list)


def zone_exit(zone: int) -> SlotCoord:
    """Bir zone'un sevkiyat çıkış (rıhtım) ağzı referansı: o zone'da (bay=0, layer=0).

    Retrieval ve rehandling hamlelerinin mesafesi bu çıkış/kenara-koyma noktasına
    göre hesaplanır.
    """
    return SlotCoord(zone, 0, 0)


def dispatch_order(state: WarehouseState, order: Order) -> DispatchResult:
    """Bir siparişin depoda bulunan (STORED) bobinlerini alır; rehandling'i sayar.

    Algoritma (docs/03 §6, "kazı ve yerine istifle" varyantı): her hedef-içeren
    sütunda, en alttaki hedefin üstündeki tüm bobinler tepeden tabana sökülür.
    Sökülenlerden hedef olanlar araç çıkışına götürülür (üretken); hedef olmayanlar
    (engelleyiciler) kenara alınıp hedef(ler) çekildikten sonra aynı sütuna geri
    istiflenir ve her biri bir rehandling sayılır. En alttaki hedefin altındaki
    bobinlere dokunulmaz.

    YAN ETKİ: ``state`` üzerinde bobinler alınır/taşınır.
    Dönüş: DispatchResult (rehandling, mesafe, yükleme süresi, alınan sayısı).
    """
    target_ids = set(order.coil_ids)
    result = DispatchResult()

    # Depoda bulunan hedef bobinlerin sütunlarını ve o sütundaki en alt hedef katını bul.
    lowest_target_by_column: dict[tuple[int, int], int] = {}
    for zone in range(state.layout.n_zones):
        for bay in range(state.layout.n_bays):
            target_layers = [
                layer
                for layer in range(state.layout.n_layers)
                if (coil := state.coil_at(SlotCoord(zone, bay, layer))) is not None
                and coil.coil_id in target_ids
            ]
            if target_layers:
                lowest_target_by_column[(zone, bay)] = min(target_layers)

    for (zone, bay), lowest in lowest_target_by_column.items():
        exit_slot = zone_exit(zone)
        height = state.stack_height(zone, bay)

        # En alttaki hedefe kadar tepeden tabana sök (layer height-1 .. lowest).
        removed: list = []  # tepeden başlayarak sökülen (coil, eski_slot) çiftleri
        for layer in range(height - 1, lowest - 1, -1):
            slot = SlotCoord(zone, bay, layer)
            coil = state.coil_at(slot)
            state.remove(coil)
            removed.append((coil, slot))

        # Sökülenleri işle: hedefler çıkışa gider, engelleyiciler geri istiflenmek üzere ayrılır.
        blockers: list = []
        for coil, slot in removed:  # tepeden taban sırası
            if coil.coil_id in target_ids:
                dist = crane_distance(slot, exit_slot)
                result.distance_m += dist
                result.loading_time_min += crane_move_time(dist)
                result.n_retrieved += 1
                result.moves.append(("retrieve", coil.coil_id, slot, exit_slot))
            else:
                blockers.append((coil, slot))
                result.rehandling_count += 1

        # Engelleyicileri aynı sütuna, tabandan başlayarak geri istifle. Orijinal
        # göreli sıraları korunduğu için ağırlık kuralı (ağır alta) kendiliğinden sağlanır.
        blockers.reverse()  # tabandan tepeye (orijinal artan-kat) sıraya getir
        next_layer = lowest
        for coil, old_slot in blockers:
            new_slot = SlotCoord(zone, bay, next_layer)
            # İki hamleli maliyet: eski konum -> kenar (çıkış) -> yeni konum.
            out_dist = crane_distance(old_slot, exit_slot)
            back_dist = crane_distance(exit_slot, new_slot)
            result.distance_m += out_dist + back_dist
            result.loading_time_min += crane_move_time(out_dist) + crane_move_time(back_dist)
            state.place(coil, new_slot)
            # Görselleştirme için işaretle: bu bobin sevkiyatta yer değiştirdi.
            coil.rehandled = True
            result.moves.append(("rehandle", coil.coil_id, old_slot, new_slot))
            next_layer += 1

    return result
