"""GEÇİCİ — Raf senaryosu poster grafiği (vinç mesafesi). Bitince silinecek.

30 held-out senaryonun ölçüm sonuçlarından iki panelli poster görseli üretir:
(A) ortalama vinç mesafesi bar grafiği (±std), (B) 30 senaryonun dağılımı (box).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# _rack_eval.py çıktısı (PPO, Heuristic, Random) — 30 görülmemiş senaryo
DATA = np.array([
    [26202, 29091, 35802], [31593, 32415, 48465], [23961, 25644, 35334],
    [28794, 28500, 41589], [25065, 26472, 36414], [24429, 27534, 38559],
    [23184, 25497, 38022], [24150, 25290, 35400], [26172, 27207, 39894],
    [20709, 23778, 34377], [23757, 25572, 38472], [31404, 33801, 45042],
    [27096, 28116, 36141], [26211, 27933, 42036], [28425, 28725, 43338],
    [26073, 27453, 39486], [22773, 26010, 35430], [26448, 27858, 43326],
    [31212, 32172, 41346], [25905, 27216, 41151], [28902, 30156, 41280],
    [27513, 27339, 34680], [32775, 31824, 44433], [24690, 27147, 38580],
    [23439, 24477, 35154], [32697, 34047, 46527], [33399, 35013, 47556],
    [26889, 26895, 35295], [23013, 26046, 38283], [22626, 25062, 37224],
], dtype=float)

ppo, heu, rnd = DATA[:, 0], DATA[:, 1], DATA[:, 2]
means = [rnd.mean(), heu.mean(), ppo.mean()]
stds = [rnd.std(), heu.std(), ppo.std()]
labels = ["Random", "Heuristic", "PPO (raf)"]
colors = ["#9aa0a6", "#4a90d9", "#2e9e5b"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2))
fig.suptitle("Raf Senaryosu (tek kat · saf rota optimizasyonu) — Vinç Mesafesi\n"
             "30 görülmemiş senaryo · düşük = iyi", fontsize=13, fontweight="bold")

# --- Panel A: ortalama bar ---
x = np.arange(3)
bars = axA.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="black", linewidth=0.7)
axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=11)
axA.set_ylabel("Ortalama vinç mesafesi (m)", fontsize=11)
axA.set_title("(A) Ortalama ± std", fontsize=11)
for b, m in zip(bars, means):
    axA.text(b.get_x() + b.get_width() / 2, m + 600, f"{m:,.0f}".replace(",", "."),
             ha="center", va="bottom", fontsize=11, fontweight="bold")
# PPO kazanım okları
axA.annotate(f"%{100*(means[0]-means[2])/means[0]:.0f} ↓\nvs Random", xy=(2, means[2]),
             xytext=(1.55, means[0]*0.9), fontsize=9, color="#2e9e5b", ha="center")
axA.set_ylim(0, max(means) * 1.18)

# --- Panel B: 30 senaryo dağılımı ---
bp = axB.boxplot([rnd, heu, ppo], labels=labels, patch_artist=True, widths=0.55,
                 medianprops=dict(color="black"))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.65)
# Her senaryoyu nokta olarak serp
for i, arr in enumerate([rnd, heu, ppo]):
    axB.scatter(np.full_like(arr, i + 1) + np.random.uniform(-0.08, 0.08, len(arr)),
                arr, s=12, color="black", alpha=0.35, zorder=3)
axB.set_ylabel("Vinç mesafesi (m)", fontsize=11)
axB.set_title("(B) 30 senaryonun dağılımı", fontsize=11)

wins = int((ppo < heu).sum())
axB.text(0.5, 0.02,
         f"PPO, Heuristic'i {wins}/30 senaryoda yendi  ·  "
         f"PPO %+{100*(heu.mean()-ppo.mean())/heu.mean():.1f} daha iyi  ·  Wilcoxon p<0.001",
         transform=axB.transAxes, ha="center", va="bottom", fontsize=9.5,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf6ee", edgecolor="#2e9e5b"))

plt.tight_layout(rect=[0, 0, 1, 0.93])
out = Path("runs/evaluation/rack_crane_distance.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Kaydedildi: {out.resolve()}")
print(f"Ortalama  PPO={ppo.mean():.0f}  Heuristic={heu.mean():.0f}  Random={rnd.mean():.0f}")
print(f"PPO Heuristic'i {wins}/30 yendi, %+{100*(heu.mean()-ppo.mean())/heu.mean():.1f}")
