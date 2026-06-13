"""RandomPolicy — en basit baseline (alt sınır).

Geçerli konumlar arasından seed'li bir generator ile düzgün dağılımla birini
seçer. Akıllı hiçbir şey yapmaz; diğer politikaların ne kadar değer kattığını
ölçmek için referans alt sınırdır.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.domain import SlotCoord, SteelCoil

from .base import PlacementPolicy

if TYPE_CHECKING:
    from src.simulation.simulator import WarehouseSimulator


class RandomPolicy(PlacementPolicy):
    """Geçerli konumlar arasından rastgele (düzgün) seçim yapan baseline."""

    def __init__(self, seed: int = 0) -> None:
        # Politikaya özel generator: tekrarlanabilirlik için (aynı seed -> aynı dizi).
        self._rng = np.random.default_rng(seed)

    def decide(self, coil: SteelCoil, sim: "WarehouseSimulator") -> SlotCoord:
        """sim.valid_actions() içinden rastgele bir konum döndürür."""
        valid = sim.valid_actions()
        if not valid:
            raise RuntimeError("Geçerli konum yok; çekirdek taşmayı yönetmeli.")
        return valid[int(self._rng.integers(0, len(valid)))]

    @property
    def name(self) -> str:
        return "Random"
