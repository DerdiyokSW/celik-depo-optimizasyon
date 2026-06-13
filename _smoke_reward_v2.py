"""GEÇİCİ — Reward v2 davranış doğrulaması. Bitince silinecek.

Kısa bir eğitim (150k, tek kademe Orta, havuz) yapar (geçici model yoluna), sonra
ortaya çıkan modelin YERLEŞTİRME DESENİNİ analiz eder: layer-0 oranı ve ortalama bay
(kapı uzaklığı). Eski PPO %100 layer-0 / ort_bay 13.2 (dejenere yayma) idi. v2 ödülü
çalışıyorsa: layer-0 oranı DÜŞMELİ (istif başlar) ve ort_bay DÜŞMELİ (kapıya yakın).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from src.simulation.loaders import load_pool
from src.simulation.event_generator import EventGenerator
from src.simulation.simulator import WarehouseSimulator
from src.ml.delay_model import DelayPredictor
from src.policies import PPOPolicy, HeuristicPolicy
from src.policies.scoring import planned_urgency
from src.rl.train_ppo import train

PROJECT_ROOT = Path(__file__).resolve().parent


def analyze(pol, sc, name):
    sim = WarehouseSimulator(sc.coils, sc.orders, sc.layout, [],
                             EventGenerator(12.0, seed=2000), seed=2000,
                             horizon_hours=24.0, vehicles=sc.vehicles)
    sim.reset()
    recs = []
    while True:
        coil = sim.pending_coil()
        if coil is None:
            break
        valid = sim.valid_actions()
        if not valid:
            sim._pending.popleft(); continue
        slot = pol.decide(coil, sim)
        recs.append((slot.bay, slot.layer))
        sim.apply_placement(slot)
    recs = np.array(recs)
    print(f"{name:<16} reh={sim.metrics.rehandling_count:>3} | "
          f"layer0=%{(recs[:,1]==0).mean()*100:>5.1f} | ort_bay={recs[:,0].mean():>5.1f} | "
          f"vinc_mesafe={sim.metrics.total_crane_distance_m:>7.0f} | n={len(recs)}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    train_pool = load_pool(PROJECT_ROOT / "data" / "pool" / "train")
    test_sc = load_pool(PROJECT_ROOT / "data" / "pool" / "test")[0]
    dm = DelayPredictor.load(str(PROJECT_ROOT / "models" / "delay_model.txt"))

    with tempfile.TemporaryDirectory() as tmp:
        mp = Path(tmp) / "ppo_v2_smoke"
        print(">>> Reward v2 ile 150k smoke egitimi (tek kademe Orta, havuz)...")
        train(total_timesteps=150_000, n_envs=4, use_curriculum=False,
              scenario_pool=train_pool, delay_model=dm, use_eval=False,
              use_terminal_reward=True, model_path=mp)
        print("\n>>> YERLESTIRME DESENI (gorulmemis senaryo):")
        analyze(PPOPolicy(str(mp.with_suffix(".zip")), delay_model=dm), test_sc, "PPO(v2-150k)")
        analyze(PPOPolicy(str(PROJECT_ROOT / "models" / "ppo_best" / "best_model.zip"),
                          delay_model=dm), test_sc, "PPO(v1-eski)")
        analyze(HeuristicPolicy(), test_sc, "Heuristic")
        print("\nBEKLENTI: v2'de layer0%% DUSMELI (istif basliyor) + ort_bay DUSMELI (kapiya yakin)")
        print("          v1: %100 layer0, ort_bay ~13 (dejenere yayma)")


if __name__ == "__main__":
    main()
