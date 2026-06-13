"""GEÇİCİ — Güncel sim görüntüsü: ana senaryoda depo doluluk deseni (Heuristic vs PPO).

Üstten görünüm (zone × bay) istif yüksekliği ısı haritası. Heuristic'in kapıya yakın kompakt
istifi vs PPO'nun yayılan/tek-kat deseni → vinç mesafesi farkını görsel açıklar. Bitince silinecek.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from src.simulation.loaders import load_pool
from src.simulation.event_generator import EventGenerator
from src.simulation.simulator import WarehouseSimulator
from src.ml.delay_model import DelayPredictor
from src.policies import HeuristicPolicy, PPOPolicy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(".").resolve()
sc = load_pool(ROOT / "data" / "pool" / "test")[0]
dm = DelayPredictor.load(str(ROOT / "models" / "delay_model.txt"))
nz, nb, nl = sc.layout.n_zones, sc.layout.n_bays, sc.layout.n_layers


def run_and_grid(policy):
    """Ana senaryoda (2 kat + affinity) sim koşar, son durumun istif-yükseklik gridini döndürür."""
    sim = WarehouseSimulator(
        sc.coils, sc.orders, sc.layout, sc.initial_placements,
        EventGenerator(12.0, seed=2000), seed=2000, horizon_hours=24.0, vehicles=sc.vehicles,
    )  # varsayılan: enforce_affinity=True, single_layer=False
    sim.run(policy, 24.0)
    g = np.zeros((nz, nb))
    for z in range(nz):
        for b in range(nb):
            g[z, b] = sim.state.stack_height(z, b)
    return g, sim.state.fill_ratio()


gh, fh = run_and_grid(HeuristicPolicy())
gp, fp = run_and_grid(PPOPolicy(str(ROOT / "models" / "ppo_best" / "best_model.zip"), delay_model=dm))

# 0=boş, 1=tek kat, 2=istifli — ayrık renkler
cmap = ListedColormap(["#f2f2f2", "#4a90d9", "#e8743b"])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

fig, axes = plt.subplots(2, 1, figsize=(13, 6.2))
fig.suptitle("Ana Senaryo — Depo Doluluk Deseni (üstten görünüm · 8 zone × 36 bay × 2 kat)\n"
             "Sol kenar = sevkiyat kapıları (bay 0) · Simülasyon: 24 saat, held-out senaryo",
             fontsize=12, fontweight="bold")

for ax, g, f, name, mean_h in [
    (axes[0], gh, fh, "Heuristic", gh[gh > 0].mean() if (gh > 0).any() else 0),
    (axes[1], gp, fp, "PPO (ana senaryo modeli)", gp[gp > 0].mean() if (gp > 0).any() else 0),
]:
    im = ax.imshow(g, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks(range(nz)); ax.set_yticklabels([f"Z{z}" for z in range(nz)], fontsize=8)
    ax.set_xticks(range(0, nb, 3)); ax.set_xticklabels(range(0, nb, 3), fontsize=8)
    ax.set_xlabel("bay (kapıdan uzaklık →)", fontsize=9)
    n2 = int((g == 2).sum())
    ax.set_title(f"{name}  ·  doluluk %{100*f:.0f}  ·  istifli kolon (2 kat) = {n2}  ·  "
                 f"ort. istif = {mean_h:.2f}", fontsize=10, loc="left")
    ax.axvline(-0.5, color="#c0392b", lw=3)  # kapı kenarı
    for s in range(nb + 1):
        ax.axvline(s - 0.5, color="white", lw=0.3)
    for s in range(nz + 1):
        ax.axhline(s - 0.5, color="white", lw=0.3)

# Ortak renk açıklaması
from matplotlib.patches import Patch
legend = [Patch(facecolor="#f2f2f2", edgecolor="gray", label="boş"),
          Patch(facecolor="#4a90d9", label="1 kat (tek)"),
          Patch(facecolor="#e8743b", label="2 kat (istifli)")]
fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.02, 1, 0.92])
out = ROOT / "runs" / "evaluation" / "sim_state.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Kaydedildi: {out}")
print(f"Heuristic: doluluk %{100*fh:.0f}, istifli kolon={(gh==2).sum()}, ort_istif={gh[gh>0].mean():.2f}")
print(f"PPO      : doluluk %{100*fp:.0f}, istifli kolon={(gp==2).sum()}, ort_istif={gp[gp>0].mean():.2f}")
