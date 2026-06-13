"""Depo geometrisi modelleri: ``SlotCoord`` ve ``WarehouseLayout``.

docs/01 §3-§4 ile birebir uyumludur. Depo 4 zone × 12 bay × 3 layer = 144
konumdan oluşur. ``SlotCoord`` tek bir konumu, ``WarehouseLayout`` deponun
değişmez yapısını (hangi zone hangi hatta hizmet eder, zone tonaj limitleri)
temsil eder.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import LogisticsLine


@dataclass(frozen=True)
class SlotCoord:
    """Depodaki tek bir istif konumu (zone, bay, layer).

    ``frozen=True`` çünkü bir konum bir değer nesnesidir: oluşturulduktan sonra
    değişmez ve hash'lenebilir olması gerekir (sözlük anahtarı / küme elemanı
    olarak istif durumunu tutarken kullanılır).

    Alanlar:
        zone: Bölge indeksi (0..n_zones-1). Lojistik hattına göre gruplanır.
        bay: Bölge içindeki sıra indeksi (0..n_bays-1).
        layer: Dikey kat (0 = zemin). Yukarı çıktıkça istif kuralları sıkılaşır.
    """

    zone: int
    bay: int
    layer: int


@dataclass
class WarehouseLayout:
    """Deponun değişmez fiziksel tanımı. ``warehouse_config.json``den yüklenir.

    Alanlar:
        n_zones: Toplam bölge sayısı (varsayılan 4).
        n_bays: Zone başına sıra sayısı (varsayılan 12).
        n_layers: Stack başına maksimum kat (varsayılan 3).
        zone_logistics: Her zone indeksini hizmet ettiği sevkiyat hattına eşler.
        zone_max_weight_ton: Her zone'un toplam tonaj limiti (kapasite kuralı).
        entry_point: Üretim bandı çıkış referansı (bay, zone) — mesafe hesabı için.
    """

    n_zones: int
    n_bays: int
    n_layers: int
    zone_logistics: dict[int, LogisticsLine]
    zone_max_weight_ton: dict[int, float]
    entry_point: tuple[int, int]

    def total_slots(self) -> int:
        """Depodaki toplam fiziksel konum sayısını döndürür (zone × bay × layer)."""
        return self.n_zones * self.n_bays * self.n_layers

    def is_valid_coord(self, coord: SlotCoord) -> bool:
        """Verilen konumun depo sınırları içinde olup olmadığını denetler.

        docs/01 doğrulama kuralı 3'ün (konum geçerliliği) tek noktadan uygulaması.
        """
        return (
            0 <= coord.zone < self.n_zones
            and 0 <= coord.bay < self.n_bays
            and 0 <= coord.layer < self.n_layers
        )

    def to_dict(self) -> dict:
        """Layout'u JSON'a yazılabilir saf-sözlük biçimine çevirir.

        JSON anahtarları metin olmak zorunda olduğu için zone indeksleri ``str``e
        çevrilir; enum üyeleri ise metinsel değerlerine indirgenir.
        """
        return {
            "n_zones": self.n_zones,
            "n_bays": self.n_bays,
            "n_layers": self.n_layers,
            # int anahtarlar JSON için str'e, LogisticsLine üyeleri metne çevrilir
            "zone_logistics": {
                str(zone): line.value for zone, line in self.zone_logistics.items()
            },
            "zone_max_weight_ton": {
                str(zone): weight for zone, weight in self.zone_max_weight_ton.items()
            },
            "entry_point": list(self.entry_point),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WarehouseLayout":
        """JSON'dan okunan saf sözlüğü ``WarehouseLayout`` nesnesine geri çevirir.

        ``to_dict``in tersidir: str anahtarlar tekrar int'e, metinler enum'a,
        liste tekrar tuple'a dönüştürülür.
        """
        return cls(
            n_zones=data["n_zones"],
            n_bays=data["n_bays"],
            n_layers=data["n_layers"],
            zone_logistics={
                int(zone): LogisticsLine(line)
                for zone, line in data["zone_logistics"].items()
            },
            zone_max_weight_ton={
                int(zone): float(weight)
                for zone, weight in data["zone_max_weight_ton"].items()
            },
            entry_point=tuple(data["entry_point"]),
        )
