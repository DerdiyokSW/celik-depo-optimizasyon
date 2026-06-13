"""WarehouseEnv — simülasyon çekirdeğini saran gymnasium ortamı (docs/07 §4-§8).

Çekirdek zaten ``pending_coil``/``valid_actions``/``apply_placement`` ilkellerini
sunduğu için ortam incedir: bir 'adım' = bir bobinin yerleştirilmesi, bir 'bölüm'
= bir simülasyon senaryosu. Geçersiz eylemler action masking ile elenir
(MaskablePPO ``action_masks()`` çağırır), bu yüzden ödülde fizik ihlali cezası yoktur.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.ml.delay_model import DelayPredictor
from src.policies.random_policy import RandomPolicy
from src.simulation.event_generator import EventGenerator
from src.simulation.loaders import Scenario
from src.simulation.simulator import WarehouseSimulator

from .action_space import action_space_size, index_to_slot, slot_to_index
from .observation import GLOBAL_DIM, PENDING_DIM, build_observation, warehouse_shape
from .reward import guidance_reward, realized_reward, terminal_reward


@dataclass
class EnvConfig:
    """WarehouseEnv'in tüm yapılandırması.

    Alanlar:
        scenario: Tek senaryo (layout kaynağı + havuz verilmezse geriye-uyumlu kullanım).
        scenario_pool: GENELLEŞTİRME için senaryo DAĞILIMI (her episode farklı popülasyon).
            Verilirse ``scenario`` yerine bundan örneklenir — overfitting'i kıran ana
            değişiklik. None ise ``[scenario]`` tek-elemanlı havuz olarak kullanılır.
        delay_model: Gözleme tahmini gecikme koymak için (hibrit mimari); None olabilir.
        event_rate_per_hour: Olay yoğunluğu (curriculum kademesi bunu ayarlar).
        horizon_hours: Bir bölümün simülasyon ufku.
        max_steps: Güvenlik için bölüm başına maksimum adım (truncation).
        base_seed: Episode tohum tabanı.
        n_seed_pool: (Eski; havuz örneklemede kullanılmaz) geriye-uyumluluk için tutulur.
        use_terminal_reward: Bölüm sonu baseline kıyas ödülü etkin mi.
        verbose_scenarios: True ise ilk tur boyunca "episode -> scenario_idx" loglar
            (smoke doğrulaması için; gerçek eğitimde False).
    """

    scenario: Scenario
    delay_model: DelayPredictor | None = None
    event_rate_per_hour: float = 12.0
    horizon_hours: float = 24.0
    max_steps: int = 600
    base_seed: int = 0
    n_seed_pool: int = 64
    use_terminal_reward: bool = True
    scenario_pool: list[Scenario] | None = None
    verbose_scenarios: bool = False
    # Senaryo modu (PPO "raf benzeri" senaryosu için): tek kat + affinity kapalı.
    enforce_affinity: bool = True
    single_layer: bool = False


class WarehouseEnv(gym.Env):
    """Çelik bobin yerleştirmeyi pekiştirmeli öğrenme ortamı olarak sunar."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig) -> None:
        super().__init__()
        self.config = config
        # GENELLEŞTİRME: tek senaryo yerine senaryo HAVUZU. Havuz verilmezse tek
        # senaryoyu 1 elemanlı havuz olarak sar (geriye-uyumlu).
        self._pool: list[Scenario] = config.scenario_pool or [config.scenario]
        self.layout = self._pool[0].layout
        self.delay_model = config.delay_model
        self.n_actions = action_space_size(self.layout)

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict(
            {
                "warehouse": spaces.Box(0.0, 1.0, warehouse_shape(self.layout), dtype=np.float32),
                "pending_coil": spaces.Box(0.0, 1.0, (PENDING_DIM,), dtype=np.float32),
                "global": spaces.Box(0.0, 1.0, (GLOBAL_DIM,), dtype=np.float32),
            }
        )

        # Her episode'da havuzdan seçilen senaryoyla simülatör YENİDEN kurulur
        # (bobin/sipariş popülasyonu değişir = genelleştirme). İlk sim havuzun ilk
        # senaryosuyla kurulur (gözlem/space ve ilk reset öncesi geçerli durum için).
        self._scenario_idx = 0
        self._sim = self._build_sim(self._pool[0], config.base_seed)
        # Terminal ödül baseline'ı POPÜLASYON-BAŞINA önbelleklenir (senaryo idx anahtarı).
        # v2: baseline = rastgele yerleşimin TOPLAM VİNÇ MESAFESİ (rehandling değil).
        self._baseline_cache: dict[int, float] = {}
        self._baseline_cost = 0.0

        self._episode = 0
        self._steps = 0

    def _build_sim(self, scenario: Scenario, episode_seed: int) -> WarehouseSimulator:
        """Verilen senaryo ve olay tohumuyla yeni bir simülatör kurar (deepcopy izolasyonu).

        Havuz örneklemede her episode bunu çağırır: farklı bobin/sipariş popülasyonu
        = ajan tek bir senaryoyu ezberleyemez. Layout tüm senaryolarda aynı olduğundan
        gözlem/aksiyon uzayı değişmez.
        """
        return WarehouseSimulator(
            scenario.coils, scenario.orders, self.layout,
            scenario.initial_placements,
            EventGenerator(self.config.event_rate_per_hour, seed=episode_seed),
            seed=episode_seed, horizon_hours=self.config.horizon_hours,
            vehicles=scenario.vehicles,
            enforce_affinity=self.config.enforce_affinity,
            single_layer=self.config.single_layer,
        )

    # --------------------------------------------------------------- gym API
    def reset(self, seed: int | None = None, options=None):
        """Havuzdan bir senaryo SEÇİP simülatörü o popülasyonla yeniden kurar.

        ``seed`` verilirse hem olay tohumu hem senaryo seçimi o tohumdan türetilir
        (deterministik eval). Verilmezse episode sayacıyla havuz dönüşümlü taranır ve
        olay tohumu artar — her episode FARKLI bobin popülasyonu + farklı olay akışı
        (geniş dağılıma maruz kalma = genelleme; ezber kırılır).
        """
        super().reset(seed=seed)
        if seed is not None:
            episode_seed = seed
            scenario_idx = seed % len(self._pool)
        else:
            scenario_idx = self._episode % len(self._pool)
            episode_seed = self.config.base_seed + self._episode
            self._episode += 1

        if self.config.verbose_scenarios and self._episode <= len(self._pool):
            print(f"[env] episode {self._episode} -> scenario_idx {scenario_idx} "
                  f"(olay tohumu {episode_seed})")

        self._scenario_idx = scenario_idx
        self._steps = 0
        scenario = self._pool[scenario_idx]
        # Simülatörü seçilen popülasyonla YENİDEN kur (overfitting fix).
        self._sim = self._build_sim(scenario, episode_seed)

        if self.config.use_terminal_reward:
            self._baseline_cost = self._baseline_distance(scenario_idx, scenario)

        self._advance_to_placeable()
        observation = build_observation(self._sim, self.delay_model)
        info = {"action_mask": self.action_masks(), "scenario_idx": scenario_idx}
        return observation, info

    def step(self, action: int):
        """Bir yerleştirme adımı: eylem -> SlotCoord -> apply_placement -> ödül."""
        self._steps += 1
        valid = self._sim.valid_actions()

        if not valid:
            # Güvenlik: yerleştirilebilir bobin yok (normalde _advance_to_placeable
            # bunu engeller). Bekleyeni düşür, ilerle.
            self._sim.skip_pending()
            self._advance_to_placeable()
            return self._build_step_return(0.0)

        coil = self._sim.pending_coil()
        valid_set = set(valid)
        desired = index_to_slot(int(action), self.layout)
        # Maskeleme normalde geçersiz eylem sunmaz; yine de güvenli tarafta kal.
        slot = desired if desired in valid_set else valid[0]

        guidance = guidance_reward(coil, slot, self._sim)
        result = self._sim.apply_placement(slot)
        # v3: doğrudan yerleştirme maliyeti (slot'tan hesaplanır) + rehandling.
        reward = guidance + realized_reward(result.rehandling_delta, slot)

        self._advance_to_placeable()
        return self._build_step_return(reward)

    def _build_step_return(self, reward: float):
        """Ortak adım dönüşü: gözlem, ödül (+terminal), bayraklar, info."""
        terminated = self._sim.is_done()
        truncated = self._steps >= self.config.max_steps
        if terminated and self.config.use_terminal_reward:
            reward += terminal_reward(self._baseline_cost, self._sim.metrics.total_crane_distance_m)
        observation = build_observation(self._sim, self.delay_model)
        info = {"action_mask": self.action_masks()}
        return observation, float(reward), bool(terminated), bool(truncated), info

    def action_masks(self) -> np.ndarray:
        """MaskablePPO'nun çağırdığı metot: geçerli konumlar True (144 uzunlukta).

        ``sim.valid_actions()`` ile birebir tutarlıdır. Hiç geçerli eylem yoksa
        (bitmiş/dolu durum) sb3'ün çökmemesi için 0. indeks True bırakılır.
        """
        mask = np.zeros(self.n_actions, dtype=bool)
        for slot in self._sim.valid_actions():
            mask[slot_to_index(slot, self.layout)] = True
        if not mask.any():
            mask[0] = True  # dejenere durum güvenlik ağı (bölüm zaten terminal)
        return mask

    def _advance_to_placeable(self) -> None:
        """Yerleştirilemeyen bobinleri düşürerek, kararı verilebilir bir duruma ilerler.

        Böylece ajan asla geçerli eylemi olmayan bir durumla karşılaşmaz (action
        masking için en az bir True garanti edilir) — ya yerleştirilebilir bir
        bobin hazırdır ya da bölüm bitmiştir.
        """
        while not self._sim.is_done():
            if self._sim.pending_coil() is None:
                break
            if self._sim.valid_actions():
                break
            self._sim.skip_pending()

    def _baseline_distance(self, scenario_idx: int, scenario: Scenario) -> float:
        """Bir POPÜLASYON için rastgele yerleşimin TOPLAM VİNÇ MESAFESİNİ hesaplar (önbellekli).

        Terminal ödülün referansıdır (v2). Senaryo başına BİR kez (sabit olay tohumuyla)
        koşulup önbelleklenir. Toplam mesafe; yerleştirme + sevkiyat + rehandling tüm
        hamlelerini kapsar — gerçek operasyon maliyetinin birleşik ölçüsüdür.
        """
        if scenario_idx in self._baseline_cache:
            return self._baseline_cache[scenario_idx]
        seed = self.config.base_seed + scenario_idx
        baseline_sim = self._build_sim(scenario, seed)
        distance = baseline_sim.run(
            RandomPolicy(seed=seed), self.config.horizon_hours
        ).total_crane_distance_m
        self._baseline_cache[scenario_idx] = distance
        return distance
