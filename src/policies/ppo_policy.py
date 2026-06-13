"""PPOPolicy — eğitilmiş PPO ajanını diğer politikalarla aynı arayüzden sunan sınıf.

Aşama 6'da gerçeklenir: eğitilmiş MaskablePPO modeliyle, gözlem + action mask
üreterek karar verir. Böylece PPO, değerlendirme hattında (Aşama 7) üç baseline
ile birebir aynı ``PlacementPolicy`` arayüzünden kıyaslanır.

Ağır bağımlılıklar (sb3, torch) yalnızca model gerçekten kullanıldığında (lazy)
içe aktarılır; böylece ``src.policies`` paketini import etmek bunları çekmez.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.domain import SlotCoord, SteelCoil

from .base import PlacementPolicy

if TYPE_CHECKING:
    from src.ml.delay_model import DelayPredictor
    from src.simulation.simulator import WarehouseSimulator


class PPOPolicy(PlacementPolicy):
    """Eğitilmiş MaskablePPO ajanını saran yerleştirme politikası."""

    def __init__(
        self,
        model_path: str | None = None,
        delay_model: "DelayPredictor | None" = None,
        model: object | None = None,
    ) -> None:
        """``model`` verilirse önceden yüklenmiş MaskablePPO yeniden kullanılır (dosyadan
        yüklenmez). Birden çok PPOPolicy aynı modeli paylaşabilir: ``predict`` durumsuz
        (deterministik ileri geçiş) olduğundan tek-thread kullanımda güvenlidir; ağır
        model yüklemesi (torch) bir kez yapılır (örn. dashboard'da 3 controller)."""
        self._model_path = model_path
        self._delay_model = delay_model
        self._model = model
        if self._model is None and model_path is not None:
            # Lazy: sb3-contrib yalnızca model yüklenirken içe aktarılır.
            from sb3_contrib import MaskablePPO

            self._model = MaskablePPO.load(model_path)

    def decide(self, coil: SteelCoil, sim: "WarehouseSimulator") -> SlotCoord:
        """Eğitilmiş PPO modeliyle karar verir.

        1) gözlemi üret, 2) sim.valid_actions()'tan action mask üret,
        3) model.predict(obs, action_masks=mask) ile eylem indeksini al,
        4) indeksi SlotCoord'a çevir (güvenlik: geçersizse ilk geçerliye düş).
        """
        if self._model is None:
            raise NotImplementedError(
                "PPOPolicy bir model_path olmadan kullanılamaz; önce model eğitilmeli "
                "(python -m src.rl.train_ppo)."
            )
        from src.rl.action_space import action_space_size, index_to_slot, slot_to_index
        from src.rl.observation import build_observation

        observation = build_observation(sim, self._delay_model)
        valid = sim.valid_actions()
        mask = np.zeros(action_space_size(sim.layout), dtype=bool)
        for slot in valid:
            mask[slot_to_index(slot, sim.layout)] = True

        action, _ = self._model.predict(observation, action_masks=mask, deterministic=True)
        slot = index_to_slot(int(action), sim.layout)
        # Maskeleme normalde geçerli eylem garantiler; yine de güvenli tarafta kal.
        valid_set = set(valid)
        if slot in valid_set:
            return slot
        return valid[0] if valid else slot

    @property
    def name(self) -> str:
        return "PPO"
