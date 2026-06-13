"""Dashboard ile simülasyon çekirdeği arasındaki köprü (callback mantığı).

``DashboardController`` bir ``WarehouseSimulator`` örneğini sunucu tarafında tutar,
seçilen politikayı takar ve adım/akış/peak/reset komutlarını çekirdeğin ilkellerine
çevirir. Yeni hiçbir karar veya fizik mantığı içermez — Aşama 4 değerlendirme
döngüsünün görsel, adım adım hâlidir.
"""

from __future__ import annotations

from pathlib import Path

from src.ml.delay_model import DelayPredictor
from src.policies import HeuristicPolicy, MLHeuristicPolicy, RandomPolicy
from src.policies.scoring import planned_urgency
from src.simulation.event_generator import EventGenerator
from src.simulation.loaders import Scenario
from src.simulation.simulator import WarehouseSimulator

_MAX_LOG_LINES = 200


class DashboardController:
    """Simülatörü tutan ve UI komutlarını ilkellere çeviren köprü."""

    def __init__(
        self,
        scenario: Scenario,
        delay_model: DelayPredictor | None = None,
        event_seed: int = 7,
        horizon_hours: float = 48.0,
        sim_seed: int = 0,
        event_rate_per_hour: float = 12.0,
        ppo_model_path: str | None = None,
        ppo_model: object | None = None,
        ppo_rack_path: str | None = None,
        ppo_model_rack: object | None = None,
        start_empty: bool = True,
        rack_mode: bool = False,
    ) -> None:
        self.scenario = scenario
        self.delay_model = delay_model
        self._event_seed = event_seed
        self._horizon = horizon_hours
        self._sim_seed = sim_seed
        self._event_rate = event_rate_per_hour
        # Raf modu: PPO'nun doğal senaryosu (tek kat = istif yok, affinity kapalı = kapı
        # ayrımı yok, hedef = saf vinç mesafesi). Raf modeli bu kurulumla eğitildiği için
        # dashboard da öyle koşmalı; aksi hâlde model yanlış ortamda haksızca kötü görünür.
        self._rack_mode = rack_mode
        # Dashboard demosunda depo BOŞTAN başlasın (initial_placements yok) — politikanın
        # depoyu sıfırdan doldurmasını izlemek durum değişimini daha net gösterir. Bilimsel
        # eval ayrı (orada bozuk başlangıç durumu rehandling baseline'ı için kullanılır).
        self._start_empty = start_empty

        # Mevcut politikalar (ML yalnızca gecikme modeli varsa; PPO yalnızca eğitilmiş
        # model dosyası varsa eklenir — yoksa dashboard'da seçilemez).
        self.policies = {"Random": RandomPolicy(seed=0), "Heuristic": HeuristicPolicy()}
        if delay_model is not None:
            self.policies["MLHeuristic"] = MLHeuristicPolicy(delay_model)
        # PPO: ANA (2 kat) ve RAF (tek kat) modelleri ayrı tutulur; canlı senaryo geçişinde
        # (set_rack_mode) aktif olan self.policies["PPO"]'ya takılır. Modeller bir kez yüklenir.
        self._ppo_main = None
        self._ppo_rack = None
        from src.policies import PPOPolicy
        if ppo_model is not None or (ppo_model_path is not None and Path(ppo_model_path).exists()):
            self._ppo_main = PPOPolicy(ppo_model_path, delay_model=delay_model, model=ppo_model)
        if ppo_model_rack is not None or (ppo_rack_path is not None and Path(ppo_rack_path).exists()):
            self._ppo_rack = PPOPolicy(ppo_rack_path, delay_model=delay_model, model=ppo_model_rack)
        active_ppo = self._ppo_rack if (rack_mode and self._ppo_rack is not None) else self._ppo_main
        if active_ppo is not None:
            self.policies["PPO"] = active_ppo
        self.current_policy_name = "Heuristic"

        self.log: list[str] = []
        # Yapısal karar günlüğü (B2): her öğe {t, coil, zone, bay, layer, reason}.
        self.decisions: list[dict] = []
        self._build_sim()

    def _build_sim(self) -> None:
        """Yeni bir simülatör örneği kurar (reset ve ilk kurulum için).

        Depo geometrisi ``scenario.layout``'tan gelir — değerlendirme (``evaluation``)
        ile AYNI tek kaynak (data/). Böylece dashboard ve eval birebir aynı 8×36×2=576
        slotluk depoyu kullanır (D1). Ayrıca ufuk da eval ile AYNI (24s) tutulur — canlı
        koşum ile held-out eval "elma-elmaya" kıyaslanabilsin (PPO genelleşme tutarlılığı,
        overfitting düzeltmesi ADIM 4). Bu rejimde tepe doluluk ~%50 (< %85): doygunluk yok.
        """
        # start_empty ise boş depo (0 doluluk); değilse senaryonun başlangıç yerleşimi.
        initial = [] if self._start_empty else self.scenario.initial_placements
        self.sim = WarehouseSimulator(
            self.scenario.coils, self.scenario.orders, self.scenario.layout,
            initial,
            EventGenerator(self._event_rate, seed=self._event_seed),
            seed=self._sim_seed, horizon_hours=self._horizon,
            vehicles=self.scenario.vehicles,
            # Raf modunda affinity/istif kısıtları kalkar (PPO'nun rota senaryosu).
            enforce_affinity=not self._rack_mode, single_layer=self._rack_mode,
        )
        self.stamp_all_urgency()

    @property
    def policy(self):
        """Seçili politika nesnesi."""
        return self.policies[self.current_policy_name]

    def set_policy(self, name: str) -> None:
        """Çalışan politikayı değiştirir (geçerli bir ad ise)."""
        if name in self.policies:
            self.current_policy_name = name
            self._log(f"Politika seçildi: {name}")

    def set_horizon(self, horizon_hours: float) -> None:
        """Simülasyon ufkunu değiştirir ve sıfırlar (daha uzun ufuk = daha yüksek doluluk).

        Görsel demo amaçlı: 24h ~%50, 48h ~%65 doluluk. Bilimsel held-out eval ayrı
        ve hep 24h'dir; bu yalnızca canlı görünümün ne kadar uzun koştuğunu belirler.
        """
        if horizon_hours != self._horizon:
            self._horizon = float(horizon_hours)
            self.reset()
            self._log(f"Ufuk {horizon_hours:.0f} saate ayarlandı (sıfırlandı).")

    @property
    def rack_mode(self) -> bool:
        """Aktif senaryo: True=raf (tek kat, affinity yok), False=ana (2 kat + affinity)."""
        return self._rack_mode

    def set_rack_mode(self, rack: bool) -> None:
        """Senaryoyu değiştirir: raf=tek kat+affinity yok+RAF modeli, ana=2 kat+affinity+ANA model.

        PPO modelini uygun olanla canlı swap eder ve sim'i sıfırlar (yeni bayraklarla kurar).
        Demoda komut satırı yerine arayüzden tek tıkla senaryo geçişi sağlar.
        """
        if rack == self._rack_mode:
            return
        self._rack_mode = rack
        active_ppo = self._ppo_rack if (rack and self._ppo_rack is not None) else self._ppo_main
        if active_ppo is not None:
            self.policies["PPO"] = active_ppo
        self.reset()  # _build_sim yeni single_layer/enforce_affinity bayraklarıyla kurar
        self._log("Senaryo: " + ("RAF (tek kat, affinity yok)" if rack else "ANA (2 kat + affinity)"))

    def step(self) -> bool:
        """Tek bir yerleştirme adımı koşar. Dönüş: simülasyon bitti mi (done)."""
        if self.sim.is_done():
            return True
        # B3: yeniden konumlandırmayı yalnızca swap yapabilen sezgisellerde (Heuristic/
        # MLHeuristic) etkinleştir; Random/PPO için kapat. pending_coil() olayları
        # işlemeden ÖNCE ayarlanmalı (PRIORITY_CHANGE orada tetiklenir).
        self.sim.set_reposition_policy(
            self.policy if isinstance(self.policy, HeuristicPolicy) else None
        )
        coil = self.sim.pending_coil()
        if coil is None:
            self._log("Simülasyon tamamlandı.")
            return True
        valid = self.sim.valid_actions()
        if not valid:
            # Depo dolu: bu bobin yerleştirilemez, çekirdek gibi atla.
            self.sim.skip_pending()
            self._log(f"{coil.coil_id}: geçerli konum yok, atlandı (depo dolu).")
            return self.sim.is_done()
        slot = self.policy.decide(coil, self.sim)
        # Swap tespiti: yalnızca swap yapabilen sezgiseller (Heuristic/MLHeuristic) için
        # ara; Random/PPO swap yapmaz, O(n) taramayı atla. swap_reason'ın trigger_coil'i
        # bu bobin ise gerekçeyi karar günlüğüne ekle (B2).
        swap = self._find_swap_for(coil) if isinstance(self.policy, HeuristicPolicy) else None
        result = self.sim.apply_placement(slot)
        self._stamp_urgency(coil)
        suffix = f"  (+{result.rehandling_delta} rehandling)" if result.rehandling_delta else ""
        swap_suffix = ""
        if swap is not None:
            mc, r = swap
            swap_suffix = (f"  | SWAP: {mc} {tuple(r['moved_from'])}→{tuple(r['moved_to'])}"
                           f" ({r['swap_cost_m']}m<{r['alt_cost_m']}m)")
        self._log(
            f"[t={self.sim.clock:.1f}s] {coil.coil_id} -> "
            f"Z{slot.zone} B{slot.bay} L{slot.layer}{suffix}{swap_suffix}"
        )
        # Yapısal karar kaydı (B2).
        reason = (f"+{result.rehandling_delta} rehandling" if result.rehandling_delta else "yerleştirme")
        if swap is not None:
            reason += swap_suffix.replace("  | ", " · ")
        self.decisions.append({
            "t": self.sim.clock, "coil": coil.coil_id,
            "zone": slot.zone, "bay": slot.bay, "layer": slot.layer, "reason": reason,
        })
        if len(self.decisions) > _MAX_LOG_LINES:
            self.decisions = self.decisions[-_MAX_LOG_LINES:]
        return self.sim.is_done()

    def _find_swap_for(self, coil) -> tuple[str, dict] | None:
        """Bu yerleştirme adımında, verilen bobin için yer açan swap'ı bulur (varsa).

        Politikanın ``decide``'ı bir bobini taşıdıysa o bobine ``swap_reason`` yazar ve
        ``trigger_coil`` = yeni (acil) bobinin kimliğidir. Depoda bu tetikleyiciye sahip
        taşınmış bobini arar; yoksa None.
        """
        for moved in self.sim.state.stored_coils():
            r = moved.swap_reason
            if r is not None and r.get("trigger_coil") == coil.coil_id:
                return moved.coil_id, r
        return None

    def run_steps(self, n: int) -> bool:
        """Ardışık n adım koşar (otonom akış). Dönüş: simülasyon bitti mi."""
        done = False
        for _ in range(n):
            done = self.step()
            if done:
                break
        return done

    def trigger_peak(self, n_orders: int = 6) -> None:
        """Zirve senaryosunu tetikler: bir grup siparişi anında etkinleştirir."""
        activated = self.sim.inject_peak(n_orders)
        self._log(f"ZİRVE olayı: {activated} sipariş aniden etkinleştirildi.")

    def reset(self) -> None:
        """Simülasyonu, logu ve karar günlüğünü başlangıca alır."""
        self._build_sim()
        self.log = []
        self.decisions = []
        self._log("Simülasyon sıfırlandı.")

    def active_delays(self) -> list[dict]:
        """Aktif siparişlerin araçlarının ML-TAHMİNİ + GERÇEK gecikmelerini döndürür.

        Canlı gecikme panelinin verisi: her aktif sipariş için araç, hat, gecikme modelinin
        tahmini ve verideki gerçek gecikme. ML'in canlı doğruluğunu göstermek için (tahmin
        vs gerçek). En büyük gerçek gecikme üstte.
        """
        out: list[dict] = []
        for order_id, entry in self.sim._active.items():
            order = entry["order"]
            vehicle = self.sim.vehicle_of(order)
            if vehicle is None:
                continue
            predicted = float(self.delay_model.predict(vehicle)) if self.delay_model else 0.0
            out.append({
                "order": order_id, "vehicle": vehicle.vehicle_id,
                "line": vehicle.target_logistics_line.value,
                "predicted": predicted, "actual": vehicle.delay_minutes,
            })
        out.sort(key=lambda d: -d["actual"])
        return out

    def stamp_all_urgency(self) -> None:
        """Depoda bulunan tüm bobinlerin urgency_score'unu görselleştirme için günceller.

        Simülatörün İÇ kopyalarını (state.stored_coils) damgalar; dış Scenario
        nesneleri deep-copy izolasyonu nedeniyle ayrıdır ve dokunulmaz.
        """
        for coil in self.sim.state.stored_coils():
            self._stamp_urgency(coil)

    def _stamp_urgency(self, coil) -> None:
        """Bir bobinin urgency_score'unu, siparişinin planlanan sevkiyatına kalan
        süreden hesaplar (HeuristicPolicy ile aynı ölçü). Görselleştirme amaçlıdır
        (renk); karar mantığını etkilemez. urgency_score veri sözleşmesinde
        "simülasyonda hesaplanır" olarak tanımlıdır.
        """
        coil.urgency_score = planned_urgency(coil, self.sim)

    def _log(self, line: str) -> None:
        """Log konsoluna bir satır ekler (son _MAX_LOG_LINES tutulur)."""
        self.log.append(line)
        if len(self.log) > _MAX_LOG_LINES:
            self.log = self.log[-_MAX_LOG_LINES:]


class ComparisonController:
    """İki politikayı AYNI tohumlanmış senaryoda eşzamanlı koşturan kıyaslama köprüsü (C1).

    İki bağımsız ``DashboardController`` tutar (lane A / lane B); ikisi de aynı
    senaryo + aynı tohumlarla kurulur, böylece BİREBİR aynı olay akışını ve bobin
    sırasını görürler (eşleştirilmiş/paired kıyaslama). Fark yalnızca takılı
    politikadır. Adım/akış komutları her iki şeridi senkron ilerletir; politika
    değişimi her iki şeridi de t=0'a alır (adil, hizalı kıyaslama).
    """

    def __init__(
        self,
        scenario: Scenario,
        delay_model: DelayPredictor | None = None,
        ppo_model_path: str | None = None,
        event_seed: int = 7,
        horizon_hours: float = 48.0,
        sim_seed: int = 0,
        event_rate_per_hour: float = 12.0,
        ppo_model: object | None = None,
        ppo_rack_path: str | None = None,
        ppo_model_rack: object | None = None,
        rack_mode: bool = False,
    ) -> None:
        kwargs = dict(
            delay_model=delay_model, ppo_model_path=ppo_model_path, ppo_model=ppo_model,
            ppo_rack_path=ppo_rack_path, ppo_model_rack=ppo_model_rack,
            event_seed=event_seed, horizon_hours=horizon_hours,
            sim_seed=sim_seed, event_rate_per_hour=event_rate_per_hour, rack_mode=rack_mode,
        )
        # İki şerit aynı kurulum; PPO modeli bir kez yüklenip yeniden kullanılır
        # (reset, simülatörü yeniden kurar ama politika nesnelerini korur).
        self.lane_a = DashboardController(scenario, **kwargs)
        self.lane_b = DashboardController(scenario, **kwargs)
        self.policies: list[str] = list(self.lane_a.policies.keys())
        # Varsayılan: en açıklayıcı kıyas — Heuristic vs PPO (PPO yoksa son politika).
        self._policy_a = "Heuristic"
        self._policy_b = "PPO" if "PPO" in self.policies else self.policies[-1]
        self.lane_a.set_policy(self._policy_a)
        self.lane_b.set_policy(self._policy_b)

    @property
    def policy_a(self) -> str:
        """A şeridinde takılı politikanın adı (layout/dropdown varsayılanı için)."""
        return self._policy_a

    @property
    def policy_b(self) -> str:
        """B şeridinde takılı politikanın adı (layout/dropdown varsayılanı için)."""
        return self._policy_b

    def set_policy_a(self, name: str) -> None:
        """A şeridinin politikasını değiştirir ve her iki şeridi t=0'a alır (hizalı kıyas)."""
        if name in self.policies:
            self._policy_a = name
            self._restart()

    def set_policy_b(self, name: str) -> None:
        """B şeridinin politikasını değiştirir ve her iki şeridi t=0'a alır (hizalı kıyas)."""
        if name in self.policies:
            self._policy_b = name
            self._restart()

    def set_horizon(self, horizon_hours: float) -> None:
        """Her iki şeridin simülasyon ufkunu değiştirir (daha uzun = daha yüksek doluluk)."""
        self.lane_a.set_horizon(horizon_hours)
        self.lane_b.set_horizon(horizon_hours)
        # set_horizon zaten reset ediyor; politikaları tekrar tak (hizalı kalsın).
        self.lane_a.set_policy(self._policy_a)
        self.lane_b.set_policy(self._policy_b)

    @property
    def rack_mode(self) -> bool:
        """Aktif senaryo (her iki şerit ortak): True=raf (tek kat), False=ana (2 kat)."""
        return self.lane_a.rack_mode

    def set_rack_mode(self, rack: bool) -> None:
        """Her iki şeridi aynı senaryoya alır (raf/ana) ve hizalı sıfırlar (paired)."""
        self.lane_a.set_rack_mode(rack)
        self.lane_b.set_rack_mode(rack)
        self.lane_a.set_policy(self._policy_a)
        self.lane_b.set_policy(self._policy_b)

    def _restart(self) -> None:
        """Her iki şeridi sıfırlayıp seçili politikaları yeniden takar (paired başlangıç)."""
        self.lane_a.reset(); self.lane_a.set_policy(self._policy_a)
        self.lane_b.reset(); self.lane_b.set_policy(self._policy_b)

    def step(self) -> None:
        """Her iki şeridi bir adım ilerletir (senkron).

        İki şerit aynı olay akışını ve aynı bobin sırasını gördüğü için her adımda
        AYNI bobini işlerler. Düşük/orta dolulukta yerleştirme sayıları eşit kalır;
        depo doyuma yaklaşırsa bir şerit bir bobini yerleştiremeyip atlayabilir
        (diğeri yerleştirebilir), bu noktada yerleştirme sayıları bir miktar
        ayrışabilir — kıyas hâlâ anlamlıdır (aynı senaryo), yalnızca birebir
        adım hizası gevşer.
        """
        self.lane_a.step()
        self.lane_b.step()

    def run_steps(self, n: int) -> None:
        """Her iki şeridi n adım ilerletir (senkron otonom akış)."""
        self.lane_a.run_steps(n)
        self.lane_b.run_steps(n)

    def reset(self) -> None:
        """Her iki şeridi başlangıca alır (seçili politikalar korunur)."""
        self._restart()

    def trigger_peak(self, n_orders: int = 6) -> None:
        """Her iki şeride aynı zirve yükünü enjekte eder (tıkanıklık kıyası için)."""
        self.lane_a.trigger_peak(n_orders)
        self.lane_b.trigger_peak(n_orders)

    @staticmethod
    def lane_metrics(controller: DashboardController) -> dict:
        """Bir şeridin kıyaslama tablosu için metrik sözlüğünü üretir."""
        m = controller.sim.metrics
        return {
            "name": controller.current_policy_name,
            "rehandling": m.rehandling_count,
            "crane_m": m.total_crane_distance_m,
            "loading_min": m.total_loading_time_min,
            "fill": controller.sim.state.fill_ratio(),
        }
