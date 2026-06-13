"""MLHeuristicPolicy — gecikme tahmini ile beslenen sezgisel.

HeuristicPolicy ile BİREBİR aynı skorlamayı kullanır; tek fark aciliyetin
hesaplanma biçimidir. Bu, ML'in net katkısını izole etmek için bilinçli bir
tasarımdır: iki politika arasındaki tek kod farkı ``_compute_urgency`` metodudur.

Mantık: aracın gecikmesi tahmin edilir; etkin sevkiyat zamanı = planlanan_varış +
tahmini_gecikme. Geç gelmesi beklenen aracın bobini daha az acil görünür, böylece
daha derine yerleştirilebilir — yer açıp rehandling'i azaltır.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain import SteelCoil

from .heuristic_policy import HeuristicPolicy
from .scoring import coil_urgency

if TYPE_CHECKING:
    from src.ml.delay_model import DelayPredictor
    from src.simulation.simulator import WarehouseSimulator


class MLHeuristicPolicy(HeuristicPolicy):
    """Aciliyeti ML ile düzeltilmiş etkin sevkiyat zamanından hesaplayan sezgisel.

    Skorlama ve karar akışı HeuristicPolicy'den miras alınır; yalnızca aciliyet
    hesabı (``_compute_urgency``) gecikme tahminini ekleyecek şekilde override edilir.
    """

    def __init__(self, delay_model: "DelayPredictor") -> None:
        self._delay_model = delay_model
        # Araç başına gecikme tahmini önbelleği: bir aracın öznitelikleri değişmez,
        # tahmini sabittir; her bobin kararında yeniden tahmin (~ms) yerine bir kez
        # hesaplanıp saklanır (dashboard/değerlendirmede büyük hız kazancı).
        self._delay_cache: dict[str, float] = {}

    def _predicted_delay_minutes(self, vehicle) -> float:
        """Aracın tahmini gecikmesini (dakika) döndürür; araç kimliğine göre önbellekli."""
        cached = self._delay_cache.get(vehicle.vehicle_id)
        if cached is None:
            cached = float(self._delay_model.predict(vehicle))
            self._delay_cache[vehicle.vehicle_id] = cached
        return cached

    def _compute_urgency(self, coil: SteelCoil, sim: "WarehouseSimulator") -> float:
        """Aciliyeti ML ile düzeltilmiş ETKİN sevkiyata kalan süreden hesaplar.

        etkin_kalan = (planlanan_dispatch + tahmini_gecikme) - şu anki saat.
        Tahmini gecikme büyükse araç geç gelecek demektir → kalan süre uzar →
        aciliyet düşer → bobin daha derine/uzağa konabilir (yer açar, rehandling azalır).
        Klasik sezgiselden TEK farkı budur (tahmini gecikme terimi).
        """
        order = sim.order_of(coil)
        planned = sim.planned_dispatch_time(order)
        if planned is None:
            return 0.0
        vehicle = sim.vehicle_of(order)
        predicted_delay_h = (self._predicted_delay_minutes(vehicle) / 60.0) if vehicle is not None else 0.0
        hours_to_dispatch = (planned + predicted_delay_h) - sim.clock
        return coil_urgency(max(0.0, hours_to_dispatch))

    @property
    def name(self) -> str:
        return "MLHeuristic"
