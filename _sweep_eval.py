"""GEÇİCİ — Doluluk/yük taraması (affinity ZORLANMIŞ, ana senaryo). Bitince silinebilir.

8/12/20 olay/saat = düşük/orta/yüksek doluluk. Her rejimde 4 politika × 30 görülmemiş
senaryo; rehandling + vinç mesafesi + doluluk ortalaması. Soru: PPO düşük dolulukta
sezgisellere yakın/iyi mi, yüksek dolulukta mı kötüleşiyor (ekran görüntüsü örüntüsü)?
"""
import json
import sys
from pathlib import Path

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

rates = [4.0, 6.0, 8.0, 12.0, 20.0]  # düşük yükleri (4/6) ekledik → ~%25-35 doluluk bandı
out = {}
for r in rates:
    out[str(r)] = {}
    for n, p in pols.items():
        df = evaluate_policy_on_pool(p, test, 2000, r, 24.0)
        rec = {"reh": float(df["rehandling"].mean()), "reh_std": float(df["rehandling"].std()),
               "cra": float(df["crane_distance_m"].mean()), "fill": float(df["fill_ratio"].mean())}
        out[str(r)][n] = rec
        print(f"rate={r:>4} {n:<11}: reh={rec['reh']:6.2f}  cra={rec['cra']:8.0f}  doluluk=%{rec['fill']*100:.0f}",
              flush=True)

(ROOT / "runs" / "evaluation" / "sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("DONE -> runs/evaluation/sweep.json")
