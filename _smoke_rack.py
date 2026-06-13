"""GEÇİCİ — PPO 'raf benzeri' senaryo smoke. Bitince silinecek.

PPO'yu RAF MODUNDA (tek kat = istif/rehandling yok, affinity kapalı = kapı ayrımı yok,
amaç = saf vinç-mesafesi/rota optimizasyonu) kısa eğitir, sonra PPO'yu Random ve
Heuristic ile bu modda kıyaslar. Beklenti: PPO en düşük vinç mesafesini almalı (öğrenen
anticipatory yerleştirme), çünkü bu RL'in doğal çalıştığı problem tipidir.
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
from src.policies import PPOPolicy, HeuristicPolicy, RandomPolicy
from src.policies.scoring import planned_urgency
from src.rl.train_ppo import train

ROOT = Path(__file__).resolve().parent


def rack_eval(pol, sc, name):
    """Bir politikayı RAF MODUNDA koşturup vinç mesafesi + yerleştirme desenini raporlar."""
    sim = WarehouseSimulator(sc.coils, sc.orders, sc.layout, [],
                             EventGenerator(12.0, seed=2000), seed=2000, horizon_hours=24.0,
                             vehicles=sc.vehicles, enforce_affinity=False, single_layer=True)
    sim.reset()
    bays = []
    urgent_bays = []
    while True:
        coil = sim.pending_coil()
        if coil is None:
            break
        valid = sim.valid_actions()
        if not valid:
            sim._pending.popleft(); continue
        u = planned_urgency(coil, sim)
        slot = pol.decide(coil, sim)
        bays.append(slot.bay)
        if u > 0.6:
            urgent_bays.append(slot.bay)
        sim.apply_placement(slot)
    ub = np.mean(urgent_bays) if urgent_bays else float("nan")
    print(f"{name:<14} vinc_mesafe={sim.metrics.total_crane_distance_m:>7.0f} | "
          f"ort_bay={np.mean(bays):>4.1f} | ACIL_ort_bay={ub:>4.1f} | n={len(bays)}")
    return sim.metrics.total_crane_distance_m


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    pool = load_pool(ROOT / "data" / "pool" / "train")
    test_sc = load_pool(ROOT / "data" / "pool" / "test")[0]
    dm = DelayPredictor.load(str(ROOT / "models" / "delay_model.txt"))

    with tempfile.TemporaryDirectory() as tmp:
        mp = Path(tmp) / "ppo_rack"
        print(">>> PPO RAF MODU eğitimi (150k, tek kat + affinity yok, reward v3)...")
        train(total_timesteps=150_000, n_envs=4, use_curriculum=False,
              scenario_pool=pool, delay_model=dm, use_eval=False, use_terminal_reward=True,
              model_path=mp, enforce_affinity=False, single_layer=True)
        print("\n>>> RAF MODU KIYAS (görülmemiş senaryo, düşük vinç = iyi):")
        ppo = rack_eval(PPOPolicy(str(mp.with_suffix(".zip")), delay_model=dm), test_sc, "PPO(rack)")
        heu = rack_eval(HeuristicPolicy(), test_sc, "Heuristic")
        rnd = rack_eval(RandomPolicy(seed=0), test_sc, "Random")
        print()
        print(f"PPO vs Heuristic: {'PPO DAHA IYI' if ppo < heu else 'Heuristic daha iyi'} "
              f"({ppo:.0f} vs {heu:.0f})")
        print(f"PPO vs Random  : {'PPO DAHA IYI' if ppo < rnd else 'Random daha iyi'} "
              f"({ppo:.0f} vs {rnd:.0f})")
        print("BEKLENTI: PPO en dusuk vinc mesafesi + ACIL bobinler dusuk bay (kapiya yakin)")


if __name__ == "__main__":
    main()
