"""Ayrık eylem uzayı ile depo konumu arasındaki kayıpsız eşleme.

PPO'nun eylem uzayı ``Discrete(n_zones * n_bays * n_layers)``tir; her tam sayı
indeks bir ``SlotCoord``'a birebir karşılık gelir. Bu modül kasıtlı olarak
hafiftir (yalnızca domain'e bağlıdır) — hem RL ortamı hem ``PPOPolicy`` ağır
gymnasium/sb3 yükü olmadan bu eşlemeyi kullanabilsin.
"""

from __future__ import annotations

from src.domain import SlotCoord, WarehouseLayout


def action_space_size(layout: WarehouseLayout) -> int:
    """Toplam ayrık eylem sayısı = zone × bay × layer."""
    return layout.n_zones * layout.n_bays * layout.n_layers


def index_to_slot(index: int, layout: WarehouseLayout) -> SlotCoord:
    """Eylem indeksini ``SlotCoord``'a çevirir (index -> (zone, bay, layer))."""
    per_zone = layout.n_bays * layout.n_layers
    zone, remainder = divmod(index, per_zone)
    bay, layer = divmod(remainder, layout.n_layers)
    return SlotCoord(zone, bay, layer)


def slot_to_index(slot: SlotCoord, layout: WarehouseLayout) -> int:
    """``SlotCoord``'u eylem indeksine çevirir (index_to_slot'un tersi)."""
    return (
        slot.zone * (layout.n_bays * layout.n_layers)
        + slot.bay * layout.n_layers
        + slot.layer
    )
