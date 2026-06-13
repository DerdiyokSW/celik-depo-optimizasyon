"""GEÇİCİ — eğitilmiş 3M raf-modelini held-out senaryolarda kıyaslar (vinç mesafesi).

Tek kat + affinity kapalı (raf modu). Her senaryoda 3 politika AYNI event seed ile
koşar (adil kıyas). Düşük vinç mesafesi = iyi. Bitince silinecek.
"""
import sys
from pathlib import Path

import numpy as np

from src.simulation.loaders import load_pool
from src.simulation.event_generator import EventGenerator
from src.simulation.simulator import WarehouseSimulator
from src.ml.delay_model import DelayPredictor
from src.policies import PPOPolicy, HeuristicPolicy, RandomPolicy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(".").resolve()
MODEL = ROOT / "models" / "ppo_warehouse.zip"

test = load_pool(ROOT / "data" / "pool" / "test")
dm = DelayPredictor.load(str(ROOT / "models" / "delay_model.txt"))
ppo = PPOPolicy(str(MODEL), delay_model=dm)
heu = HeuristicPolicy()


def run(pol, sc, seed):
    """Bir politikayı raf modunda koşturur, toplam vinç mesafesini (m) döndürür."""
    sim = WarehouseSimulator(
        sc.coils, sc.orders, sc.layout, [],
        EventGenerator(12.0, seed=seed), seed=seed, horizon_hours=24.0,
        vehicles=sc.vehicles, enforce_affinity=False, single_layer=True,
    )
    sim.reset()
    while True:
        coil = sim.pending_coil()
        if coil is None:
            break
        if not sim.valid_actions():
            sim._pending.popleft()
            continue
        sim.apply_placement(pol.decide(coil, sim))
    return sim.metrics.total_crane_distance_m


N = len(test)
rows = []
pw = 0
for i in range(N):
    sc = test[i]
    seed = 3000 + i
    dp = run(ppo, sc, seed)
    dh = run(heu, sc, seed)
    dr = run(RandomPolicy(seed=seed), sc, seed)
    rows.append((dp, dh, dr))
    pw += int(dp < dh)
    print(f"sc{i:02d} seed{seed}: PPO={dp:8.0f}  Heu={dh:8.0f}  Rnd={dr:8.0f}  "
          f"{'PPO<Heu' if dp < dh else 'Heu<=PPO'}")

A = np.array(rows)
mp, mh, mr = A.mean(axis=0)
print("=" * 64)
print(f"ORTALAMA  PPO={mp:.0f}  Heuristic={mh:.0f}  Random={mr:.0f}  (N={N})")
print(f"PPO, Heuristic'i {pw}/{N} senaryoda yendi")
print(f"PPO vs Random   : %{100 * (mr - mp) / mr:.1f} daha az vinç mesafesi")
print(f"PPO vs Heuristic: %{100 * (mh - mp) / mh:+.1f} (pozitif = PPO daha iyi)")

# --- İstatistiksel anlamlılık: Wilcoxon işaretli-sıra (eşli) testi ---
# PPO ile Heuristic AYNI senaryolarda eşli ölçüldü; fark sistematik mi tesadüf mü?
try:
    from scipy.stats import wilcoxon
    stat, pval = wilcoxon(A[:, 0], A[:, 1])  # PPO vs Heuristic
    karar = "ANLAMLI (PPO gerçekten daha iyi)" if pval < 0.05 else "anlamsız (tesadüf olabilir)"
    print(f"Wilcoxon PPO<Heuristic: p={pval:.5f} -> {karar}")
except Exception as e:  # scipy yoksa atla
    print(f"(Wilcoxon atlandı: {e})")
