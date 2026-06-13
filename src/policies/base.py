"""PlacementPolicy — tüm yerleştirme politikalarının uyduğu ortak soyut sözleşme.

Simülasyon çekirdeği yalnızca bu arayüzü tanır; somut politikayı bilmez. Dört
politika (Random/Heuristic/MLHeuristic/PPO) bunu uygular ve birbirinin yerine
takılabilir — değerlendirme (Aşama 7) bu sayede tek hatla yapılır. Bu, projenin
"risk izolasyonu" stabilite garantisidir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.domain import SlotCoord, SteelCoil

if TYPE_CHECKING:
    # Çalışma zamanında içe aktarılmaz (döngü önleme); yalnızca tip ipucu.
    from src.simulation.simulator import WarehouseSimulator


class PlacementPolicy(ABC):
    """Tüm yerleştirme politikalarının uyduğu ortak sözleşme."""

    @abstractmethod
    def decide(self, coil: SteelCoil, sim: "WarehouseSimulator") -> SlotCoord:
        """Bekleyen bobin için bir yerleştirme konumu seçer.

        Dönen konum, ``sim.valid_actions()`` listesinden biri OLMAK ZORUNDADIR;
        politika geçerli konumlar dışına çıkamaz. Politika ``sim`` üzerinden
        ``valid_actions()``, ``state``, ``order_of(coil)`` ve ``vehicle_of(order)``
        ilkellerine erişir.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Raporlama ve grafiklerde kullanılan kısa ad (ör. 'Heuristic')."""

    def reposition_on_priority_change(
        self, coil: SteelCoil, sim: "WarehouseSimulator"
    ) -> "SlotCoord | None":
        """Aciliyeti yükselen yerleşik bir bobini gerekirse daha iyi konuma taşır (B3).

        VARSAYILAN: hiçbir şey yapma (None). Yalnızca swap yapabilen sezgiseller
        (Heuristic/MLHeuristic) bunu override eder; Random/PPO yeniden konumlandırma
        yapmaz. Bu kanca yalnızca dashboard'da (opt-in) çağrılır — değerlendirme
        hattında çağrılmaz, dolayısıyla eval sonuçlarını ETKİLEMEZ.
        """
        return None
