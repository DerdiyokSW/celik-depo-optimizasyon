"""GEÇİCİ — ANA senaryo (2 kat + affinity ZORLANMIŞ) held-out re-ölçümü. Bitince silinecek.

Eski 4.13 affinity eklenmeden ölçülmüştü. Simülatör artık varsayılan enforce_affinity=True;
bu eval, ana-senaryo modelini (models/ppo_best, affinity'siz eğitilmiş) GERÇEK kısıtlar
altında ölçer: rehandling + vinç mesafesi, PPO vs Heuristic vs Random, 30 görülmemiş pop.
Resmi comparison.json'u EZMEZ (sadece konsola yazar).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.simulation.loaders import load_pool
from src.evaluation.runner import evaluate_policy_on_pool
from src.ml.delay_model import DelayPredictor
from src.policies import PPOPolicy, HeuristicPolicy, MLHeuristicPolicy, RandomPolicy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(".").resolve()
test = load_pool(ROOT / "data" / "pool" / "test")
dm = DelayPredictor.load(str(ROOT / "models" / "delay_model.txt"))
ppo = PPOPolicy(str(ROOT / "models" / "ppo_best" / "best_model.zip"), delay_model=dm)

pols = {"PPO": ppo, "Heuristic": HeuristicPolicy(),
        "MLHeuristic": MLHeuristicPolicy(dm), "Random": RandomPolicy(seed=0)}
print(">>> ANA senaryo (2 kat + affinity ZORLANMIŞ) — 30 görülmemiş popülasyon, eşli")
res = {n: evaluate_policy_on_pool(p, test, 2000, 12.0, 24.0) for n, p in pols.items()}

for met, birim, low_good in [("rehandling", "adet", True), ("crane_distance_m", "m", True)]:
    print(f"\n=== {met} ({birim}) ===")
    for n in pols:
        a = res[n][met].to_numpy()
        print(f"  {n:<10} ort={a.mean():9.2f}  std={a.std():8.2f}")
    # PPO vs Heuristic eşli Wilcoxon + kim kaç senaryoda önde
    pp = res["PPO"][met].to_numpy()
    hh = res["Heuristic"][met].to_numpy()
    pwin = int((pp < hh).sum())
    try:
        _, pval = wilcoxon(pp, hh)
    except Exception as e:
        pval = float("nan")
    kazanan = "PPO" if pp.mean() < hh.mean() else "Heuristic"
    print(f"  -> ortalamada {kazanan} daha iyi | PPO, Heuristic'i {pwin}/30 yendi | "
          f"Wilcoxon p={pval:.5f}")
