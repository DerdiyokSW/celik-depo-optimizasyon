"""WarehouseExtractor — depo ızgarasını CNN, pending+global'i MLP ile işleyip
birleştiren özel SB3 feature extractor (Aşama 6 PPO için).

Sorun: SB3 varsayılan FlattenExtractor + 64×64 MLP, depo gözlem tensörünü
(8×36×2×3) düz vektöre çevirip minik MLP'den geçirir → ağ konumların uzamsal
komşuluğunu (aynı zone'da yan sütunlar, üst-alt katlar) temsil edemiyor.

Çözüm: Depo tensörü ``(zone, bay, layer, kanal)``, ``(kanal × kat, zone, bay)`` olarak
yeniden düzenlenip Conv2d'ye verilir; bekleyen bobin ve küresel öznitelik
vektörleri ayrı MLP'den geçer; iki temsil birleşip tek özellik vektörü üretir.
"""

from __future__ import annotations

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class WarehouseExtractor(BaseFeaturesExtractor):
    """Depo ızgarası için CNN tabanlı özellik çıkarıcı.

    Çıktısı sabit ``features_dim`` uzunluklu vektör; SB3 ``MultiInputPolicy``
    bunu politika ve değer başlıklarına besler.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        # Depo tensör şekli: (zone, bay, layer, kanal).
        z, b, l, c = observation_space["warehouse"].shape
        in_channels = l * c  # kat × kanal birleşik kanal sayısı
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        # CNN çıkış boyutunu bir kez deneme tensörüyle ölç (dense katman için).
        with th.no_grad():
            n_flatten = int(self.cnn(th.zeros(1, in_channels, z, b)).shape[1])
        extra = (
            int(observation_space["pending_coil"].shape[0])
            + int(observation_space["global"].shape[0])
        )
        self.head = nn.Sequential(
            nn.Linear(n_flatten + extra, features_dim), nn.ReLU(),
        )

    def forward(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        """Gözlem sözlüğünü tek özellik vektörüne dönüştürür."""
        wh = obs["warehouse"]  # (N, z, b, l, c)
        n = wh.shape[0]
        # (N, z, b, l, c) -> (N, l, c, z, b) -> (N, l*c, z, b)
        wh = wh.permute(0, 3, 4, 1, 2).contiguous()
        wh = wh.reshape(n, wh.shape[1] * wh.shape[2], wh.shape[3], wh.shape[4])
        spatial = self.cnn(wh)
        rest = th.cat([obs["pending_coil"], obs["global"]], dim=1)
        return self.head(th.cat([spatial, rest], dim=1))
