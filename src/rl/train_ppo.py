"""MaskablePPO eğitim betiği (docs/07 §10). Giriş noktası (main).

Çalıştırma: ``python -m src.rl.train_ppo [--total-timesteps N] [--no-curriculum]
[--resume-from PATH]``

Tekerlek yeniden icat edilmez: ortam, gözlem, ödül ve curriculum bizim; algoritma
(MaskablePPO) sb3-contrib'den hazır. Action masking sayesinde ajan yalnızca geçerli
konumlardan seçer. Eğitim curriculum kademeleri boyunca (kolay->zor) ilerler.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.ml.delay_model import DelayPredictor
from src.simulation.loaders import Scenario, load_pool

from .curriculum import STAGES, CurriculumStage
from .feature_extractor import WarehouseExtractor
from .warehouse_env import EnvConfig, WarehouseEnv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODEL_PATH: Path = PROJECT_ROOT / "models" / "ppo_warehouse"
DELAY_MODEL_PATH: Path = PROJECT_ROOT / "models" / "delay_model.txt"
TENSORBOARD_DIR: Path = PROJECT_ROOT / "runs" / "ppo_warehouse"
TRAIN_POOL_DIR: Path = PROJECT_ROOT / "data" / "pool" / "train"
# Periyodik checkpoint dizini — kesinti hâlinde buradan --resume-from ile devam.
CHECKPOINT_DIR: Path = PROJECT_ROOT / "models" / "checkpoints"
CHECKPOINT_FREQ: int = 100_000  # her ~100k adımda bir model kaydet (kurtarma güvencesi)


class _KeepAwake:
    """Eğitim boyunca Windows'un uykuya/ekran kapatmaya geçmesini engeller (context manager).

    ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` ile sisteme
    "çalışıyorum, uyuma" der; çıkışta bayrağı bırakır. Otomatik/boşta uykuyu önler
    (manuel uyku veya kapak kapatma eylemini değil — onlar ayrıca güç ayarından
    kapatılmalı). Windows dışında no-op.
    """

    def __enter__(self):
        self._ok = False
        try:
            import ctypes
            # ES_CONTINUOUS=0x80000000, ES_SYSTEM_REQUIRED=0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            self._ok = True
            print("[keep-awake] Uyku engellendi (eğitim süresince).")
        except Exception:
            pass  # Windows değil veya çağrı başarısız — sessiz geç
        return self

    def __exit__(self, *exc):
        if self._ok:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS: bayrağı bırak
            except Exception:
                pass
        return False

# docs/07 §10 başlangıç hiperparametreleri.
PPO_HYPERPARAMS: dict = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 512,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
}


def _tensorboard_log_dir() -> str | None:
    """TensorBoard kuruluysa log dizinini, değilse None döndürür (zarif geri dönüş)."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("Uyarı: tensorboard kurulu değil; TB loglaması atlanıyor.")
        return None
    return str(TENSORBOARD_DIR)


def _mask_fn(env) -> "object":
    """ActionMasker'ın çağırdığı maske fonksiyonu (ortamın action_masks'ını döndürür)."""
    return env.action_masks()


def make_env_fn(config: EnvConfig):
    """Bir WarehouseEnv'i ActionMasker + Monitor ile saran fabrika (VecEnv için).

    Monitor, bölüm ödülü/uzunluğunu kaydeder; böylece TensorBoard'da öğrenme eğrisi
    (rollout/ep_rew_mean) izlenebilir — PPO'nun öğrendiğinin görsel kanıtı.
    """

    def _init():
        return Monitor(ActionMasker(WarehouseEnv(config), _mask_fn))

    return _init


def _build_vec_env(config: EnvConfig, n_envs: int) -> DummyVecEnv:
    """n_envs adet maskelenmiş ortamdan bir DummyVecEnv kurar.

    Windows'ta SubprocVecEnv spawn/pickle zorlukları olduğundan DummyVecEnv
    (aynı süreç) tercih edilir — sağlam ve yeterince hızlı.
    """
    return DummyVecEnv([make_env_fn(config) for _ in range(n_envs)])


def train(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    use_curriculum: bool = True,
    resume_from: str | None = None,
    horizon_hours: float = 24.0,
    seed: int = 42,
    scenario: Scenario | None = None,
    scenario_pool: list[Scenario] | None = None,
    delay_model: DelayPredictor | None = None,
    use_terminal_reward: bool = True,
    use_eval: bool = True,
    model_path: str | Path = MODEL_PATH,
    verbose_scenarios: bool = False,
    enforce_affinity: bool = True,
    single_layer: bool = False,
) -> MaskablePPO:
    """MaskablePPO ajanını curriculum kademeleri boyunca eğitir ve kaydeder.

    ``scenario_pool`` verilirse her episode havuzdan FARKLI bir popülasyon örneklenir
    (overfitting'i kıran ana değişiklik); verilmezse tek ``scenario`` (eski davranış,
    ezber riski). ``resume_from`` ile warm-start (Katman 3, docs/07 §12).
    """
    if scenario_pool is None and scenario is None:
        scenario = Scenario.from_data_dir()
    # Layout kaynağı + geriye-uyumlu tek-senaryo alanı: havuz varsa ilk eleman.
    if scenario is None:
        scenario = scenario_pool[0]
    if delay_model is None and DELAY_MODEL_PATH.exists():
        delay_model = DelayPredictor.load(str(DELAY_MODEL_PATH))

    # Kademe seçimi. Resume (checkpoint'ten devam) hâlinde curriculum'u baştan
    # TEKRARLAMA — ajan kolay/orta/zor'u zaten gördü; kalan adımları tek kademede
    # (Orta 12/h, eval+canlı rejimiyle aynı) tamamla. Bu, kesinti sonrası temiz devamı sağlar.
    if resume_from is not None:
        stages = [STAGES[1]]
    elif use_curriculum:
        stages = STAGES
    else:
        stages = [STAGES[1]]  # tek kademe: Orta

    def config_for(stage: CurriculumStage) -> EnvConfig:
        return EnvConfig(
            scenario=scenario, scenario_pool=scenario_pool, delay_model=delay_model,
            event_rate_per_hour=stage.event_rate_per_hour, horizon_hours=horizon_hours,
            base_seed=seed, use_terminal_reward=use_terminal_reward,
            verbose_scenarios=verbose_scenarios,
            enforce_affinity=enforce_affinity, single_layer=single_layer,
        )

    # İlk kademenin ortamıyla modeli kur (veya warm-start ile yükle).
    tb_log = _tensorboard_log_dir()
    first_env = _build_vec_env(config_for(stages[0]), n_envs)
    # Politika ağı kurulumu: depo ızgarası için CNN feature extractor + büyük baş.
    # Varsayılan FlattenExtractor+64×64 MLP uzamsal yapıyı yakalayamıyordu (PPO
    # bu yüzden Random'ı geçemedi); CNN ile aksiyon-konum eşlemesi öğrenilebilir.
    policy_kwargs = dict(
        features_extractor_class=WarehouseExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256],
    )
    if resume_from is not None:
        model = MaskablePPO.load(resume_from, env=first_env, tensorboard_log=tb_log)
    else:
        model = MaskablePPO(
            "MultiInputPolicy", first_env, seed=seed, verbose=1,
            tensorboard_log=tb_log, policy_kwargs=policy_kwargs, **PPO_HYPERPARAMS,
        )

    # Checkpoint callback: her kademede ~CHECKPOINT_FREQ adımda bir modeli diske yazar.
    # Kesinti (uyku/çökme) hâlinde en son checkpoint'ten --resume-from ile devam edilir.
    # n_envs paralel olduğundan callback adım sayacı env-başına; save_freq'i buna böl.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, CHECKPOINT_FREQ // n_envs),
        save_path=str(CHECKPOINT_DIR), name_prefix="ppo_ckpt", verbose=1,
    )

    steps_per_stage = max(1, total_timesteps // len(stages))
    # Eğitim boyunca sistemin uykuya geçmesini engelle (boşta/otomatik uyku).
    with _KeepAwake():
        for index, stage in enumerate(stages):
            env = first_env if index == 0 else _build_vec_env(config_for(stage), n_envs)
            model.set_env(env)
            callbacks: list = [ckpt_cb]
            if use_eval:
                # Maskeyi dikkate alan değerlendirme; en iyi modeli ayrıca saklar.
                # En-iyi yolu model_path'ten türetilir (rack/ana modeller çakışmasın):
                # model_path=ppo_rack -> ppo_rack_best, ppo_warehouse -> ppo_best.
                _mp = Path(model_path)
                best_dir = _mp.parent / (_mp.stem + "_best" if _mp.stem != "ppo_warehouse" else "ppo_best")
                eval_env = _build_vec_env(config_for(stage), 1)
                callbacks.append(MaskableEvalCallback(
                    eval_env, best_model_save_path=str(best_dir),
                    eval_freq=max(2048, steps_per_stage // 4), n_eval_episodes=3, verbose=0,
                ))
            print(f"[Curriculum] Kademe '{stage.name}' ({stage.event_rate_per_hour}/saat) "
                  f"-> {steps_per_stage} adım")
            # Resume'da adım sayacını sıfırlama (log/sayaç devam etsin).
            reset_ts = (index == 0) and (resume_from is None)
            model.learn(steps_per_stage, reset_num_timesteps=reset_ts,
                        callback=CallbackList(callbacks))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"PPO modeli kaydedildi -> {model_path}.zip")
    return model


def main() -> None:
    """Komut satırı arayüzü."""
    # Windows konsolu (cp1252) Türkçe karakterleri kodlayamayabilir; çıktıyı UTF-8'e al.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="MaskablePPO depo yerleştirme eğitimi")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-eval", action="store_true")
    # Genelleştirme: train havuzundan örnekle (overfitting fix). data/pool/train.
    parser.add_argument("--pool", action="store_true",
                        help="data/pool/train senaryo havuzundan örnekle (ezberi kır)")
    parser.add_argument("--verbose-scenarios", action="store_true",
                        help="ilk tur episode->scenario_idx loglarını göster (smoke)")
    # PPO 'raf benzeri' senaryosu: tek kat (istif yok) + affinity yok (kapı ayrımı yok),
    # amaç saf vinç-mesafesi/rota optimizasyonu — RL'in doğal alanı.
    parser.add_argument("--rack-mode", action="store_true",
                        help="PPO raf senaryosu: --single-layer + --no-affinity birlikte")
    parser.add_argument("--single-layer", action="store_true", help="yalnız zemin (istif yok)")
    parser.add_argument("--no-affinity", action="store_true", help="kapı/hat ayrımı yok")
    args = parser.parse_args()
    single_layer = args.single_layer or args.rack_mode
    enforce_affinity = not (args.no_affinity or args.rack_mode)

    scenario_pool = None
    if args.pool:
        print(f"Train havuzu yükleniyor -> {TRAIN_POOL_DIR}")
        scenario_pool = load_pool(TRAIN_POOL_DIR)
        print(f"Havuzda {len(scenario_pool)} farklı popülasyon (her episode biri seçilir).")

    rack = single_layer or not enforce_affinity
    if rack:
        print(f"PPO RAF MODU: single_layer={single_layer}, affinity={'KAPALI' if not enforce_affinity else 'açık'}")
    # Rack modeli ana modeli (ppo_best, 2-kat+affinity senaryosu) EZMESİN: ayrı yola kaydet.
    model_path = MODEL_PATH.parent / ("ppo_rack" if rack else "ppo_warehouse")

    train(
        total_timesteps=args.total_timesteps, n_envs=args.n_envs,
        use_curriculum=not args.no_curriculum, resume_from=args.resume_from,
        horizon_hours=args.horizon_hours, seed=args.seed, use_eval=not args.no_eval,
        scenario_pool=scenario_pool, verbose_scenarios=args.verbose_scenarios,
        enforce_affinity=enforce_affinity, single_layer=single_layer,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
