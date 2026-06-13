"""WarehouseState — deponun 3B anlık durumu (144 konum) ve yerleştirme/sorgu API'si.

Durum, hangi konumda hangi bobinin olduğunu bir sözlükte tutar. RL bölümünde
(Aşama 6) çok sık sıfırlanıp sorgulanacağı için sorgular ucuzdur. Konumsal mantık
bobin nesnelerinin ``status``/``location`` alanlarına bağlı değildir (bir bobin
"depoda" demek, grid'de bulunması demektir); bu alanlar yine de downstream
tüketiciler (gözlem/görselleştirme) için güncel tutulur.
"""

from __future__ import annotations

import numpy as np

from src.domain import CoilStatus, SlotCoord, SteelCoil, WarehouseLayout

from .constraints import can_place


class WarehouseState:
    """Deponun anlık doluluk durumu ve üzerinde güvenli işlemler.

    İç temsil: ``(zone, bay, layer)`` anahtarlı sözlük -> o konumdaki ``SteelCoil``.
    Zone toplam ağırlıkları kapasite kontrolü için ayrıca önbelleğe alınır.
    """

    def __init__(self, layout: WarehouseLayout) -> None:
        self.layout = layout
        self._grid: dict[tuple[int, int, int], SteelCoil] = {}
        # Zone başına toplam ağırlık (kapasite kuralı için O(1) erişim).
        self._zone_weight: dict[int, float] = {z: 0.0 for z in range(layout.n_zones)}

    @staticmethod
    def _key(slot: SlotCoord) -> tuple[int, int, int]:
        """Konumu sözlük anahtarına çevirir (frozen SlotCoord da hashlenebilir ama
        tuple anahtar grid'i hafif ve kopyalaması ucuz tutar)."""
        return (slot.zone, slot.bay, slot.layer)

    def is_empty(self, slot: SlotCoord) -> bool:
        """Konum boş mu?"""
        return self._key(slot) not in self._grid

    def coil_at(self, slot: SlotCoord) -> SteelCoil | None:
        """Konumdaki bobini döndürür; boşsa None."""
        return self._grid.get(self._key(slot))

    def zone_weight(self, zone: int) -> float:
        """Bir zone'daki bobinlerin toplam ağırlığı (ton)."""
        return self._zone_weight[zone]

    def place(self, coil: SteelCoil, slot: SlotCoord) -> None:
        """Bobini verilen konuma yerleştirir.

        Çağıran, konumun geçerli olduğunu önceden ``can_place`` ile garanti etmelidir;
        yine de dolu konuma yazımı bir hata olarak yakalar.
        """
        key = self._key(slot)
        if key in self._grid:
            raise ValueError(f"Konum dolu, yerleştirilemez: {slot}")
        self._grid[key] = coil
        self._zone_weight[slot.zone] += coil.weight_ton
        # Downstream fidelity: bobinin kendi durumu da güncellenir.
        coil.location = slot
        coil.status = CoilStatus.STORED

    def remove(self, coil: SteelCoil) -> SteelCoil:
        """Bobini bulunduğu konumdan çıkarır ve geri döndürür (statüsünü çağıran ayarlar)."""
        if coil.location is None:
            raise ValueError(f"Bobinin konumu yok, çıkarılamaz: {coil.coil_id}")
        key = self._key(coil.location)
        if self._grid.get(key) is not coil:
            raise ValueError(f"Bobin beklenen konumda değil: {coil.coil_id} @ {coil.location}")
        del self._grid[key]
        self._zone_weight[coil.location.zone] -= coil.weight_ton
        coil.location = None
        return coil

    def valid_slots(self, coil: SteelCoil) -> list[SlotCoord]:
        """Bu bobinin TÜM kısıtları sağlayarak konabileceği konumlar.

        Süreklilik kuralı gereği bir kolonda yalnızca BİR SONRAKİ boş kat (stack
        yüksekliği) aday olabilir; alt katlar dolu, üst katlar süreksizdir. Bu
        yüzden her kolon için yalnızca o tek aday ``can_place`` ile sınanır —
        sonuç tüm-konum taramasıyla AYNI küme, ama ~yarı maliyet (RL/değerlendirme
        sıcak yolu). Sabit (zone, bay) sırası determinizmi korur.
        """
        result: list[SlotCoord] = []
        for zone in range(self.layout.n_zones):
            for bay in range(self.layout.n_bays):
                layer = self.stack_height(zone, bay)  # bir sonraki boş kat
                if layer >= self.layout.n_layers:
                    continue  # kolon dolu
                slot = SlotCoord(zone, bay, layer)
                if can_place(self, coil, slot):
                    result.append(slot)
        return result

    def fill_ratio(self) -> float:
        """Dolu konum / toplam konum oranı (0..1)."""
        return len(self._grid) / self.layout.total_slots()

    def stack_height(self, zone: int, bay: int) -> int:
        """Bir (zone, bay) sütununda dolu kat sayısı (süreklilik gereği = en üst dolu kat+1)."""
        height = 0
        for layer in range(self.layout.n_layers):
            if (zone, bay, layer) in self._grid:
                height += 1
        return height

    def occupied_count(self) -> int:
        """Dolu konum sayısı."""
        return len(self._grid)

    def stored_coils(self) -> list[SteelCoil]:
        """Depoda bulunan tüm bobinleri döndürür (görselleştirme/sorgu için)."""
        return list(self._grid.values())

    def snapshot(self) -> np.ndarray:
        """Durumu sayısal tensöre çevirir: (n_zones, n_bays, n_layers) ağırlık dizisi.

        Boş konum 0.0, dolu konum bobinin ağırlığıdır (ton). Aşama 6 gözlem
        uzayının temelidir; oradaki ihtiyaçlara göre zenginleştirilebilir.
        """
        grid = np.zeros(
            (self.layout.n_zones, self.layout.n_bays, self.layout.n_layers),
            dtype=np.float32,
        )
        for (zone, bay, layer), coil in self._grid.items():
            grid[zone, bay, layer] = coil.weight_ton
        return grid

    def copy(self) -> "WarehouseState":
        """Durumun ucuz kopyası: grid ve zone ağırlıkları kopyalanır.

        Not: Bobin nesneleri paylaşılır (sığ). Aşama 2 için tek-durumlu kullanımda
        sorun değildir; RL'de (Aşama 6) bağımsız bobin kopyaları gerekirse bu
        metot orada derinleştirilecektir.
        """
        clone = WarehouseState(self.layout)
        clone._grid = dict(self._grid)
        clone._zone_weight = dict(self._zone_weight)
        return clone
