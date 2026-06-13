"""GEÇİCİ — Poster grafikleri (Times New Roman, 300 DPI, gerçek ölçüm verisinden).

Veriler: held-out 30 senaryo eval çıktıları (_main_eval.py + _rack_eval.py koşumları).
Üç grafik: (1) ana senaryo 4 politika, (2) raf senaryosu 3 politika, (3) metodolojik
düzeltme yolculuğu. Tablolar PowerPoint'te kurulacak (font tutarlılığı). Bitince silinebilir.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Poster fontu: Times New Roman (şart). Yoksa serif'e düşer.
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("runs/evaluation/poster")
OUT.mkdir(parents=True, exist_ok=True)
COL = {"Random": "#9aa0a6", "Heuristic": "#3f7fc4", "MLHeuristic": "#8e6fc9", "PPO": "#2e9e5b"}
DPI = 300


def label_bars(ax, bars, vals, fmt="{:.0f}", dy=0.01):
    ymax = max(v for v in vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * dy, fmt.format(v),
                ha="center", va="bottom", fontsize=12)


# ============ FIG 1: ANA SENARYO (2 kat + affinity, 30 held-out) ============
pol = ["MLHeuristic", "Heuristic", "PPO", "Random"]
reh = {"MLHeuristic": (8.30, 2.91), "Heuristic": (8.57, 3.04), "PPO": (10.40, 5.04), "Random": (33.10, 7.92)}
cra = {"MLHeuristic": 29806, "Heuristic": 26525, "PPO": 35675, "Random": 40717}
cra_std = {"MLHeuristic": 3009, "Heuristic": 3056, "PPO": 4325, "Random": 4545}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
fig.suptitle("Ana Senaryo — Gerçekçi Depo (2 kat + affinity) · 30 görülmemiş senaryo",
             fontsize=15, fontweight="bold")
colors = [COL[p] for p in pol]

b1 = a1.bar(pol, [reh[p][0] for p in pol], yerr=[reh[p][1] for p in pol], capsize=5,
            color=colors, edgecolor="black", linewidth=0.6)
label_bars(a1, b1, [reh[p][0] for p in pol], "{:.1f}")
a1.set_ylabel("Rehandling (ort ± std)", fontsize=13)
a1.set_title("Rehandling — düşük = iyi", fontsize=13)
a1.text(0.5, 0.93, "Kazanan: Sezgisel (PPO 10.4 > 8.6, p=0.11)", transform=a1.transAxes,
        ha="center", fontsize=10.5, style="italic", color="#444")

b2 = a2.bar(pol, [cra[p] for p in pol], yerr=[cra_std[p] for p in pol], capsize=5,
            color=colors, edgecolor="black", linewidth=0.6)
label_bars(a2, b2, [cra[p] for p in pol], "{:.0f}")
a2.set_ylabel("Vinç mesafesi (m, ort ± std)", fontsize=13)
a2.set_title("Vinç mesafesi — düşük = iyi", fontsize=13)
a2.text(0.5, 0.93, "Kazanan: Heuristic (PPO 0/30, p<0.001)", transform=a2.transAxes,
        ha="center", fontsize=10.5, style="italic", color="#444")
for ax in (a1, a2):
    ax.tick_params(labelsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig1_ana_senaryo.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ============ FIG 2: RAF SENARYOSU (tek kat, rota, 30 held-out) ============
# _rack_eval.py 30 senaryo vinç mesafesi (PPO, Heuristic, Random)
RACK = np.array([
    [26202, 29091, 35802], [31593, 32415, 48465], [23961, 25644, 35334], [28794, 28500, 41589],
    [25065, 26472, 36414], [24429, 27534, 38559], [23184, 25497, 38022], [24150, 25290, 35400],
    [26172, 27207, 39894], [20709, 23778, 34377], [23757, 25572, 38472], [31404, 33801, 45042],
    [27096, 28116, 36141], [26211, 27933, 42036], [28425, 28725, 43338], [26073, 27453, 39486],
    [22773, 26010, 35430], [26448, 27858, 43326], [31212, 32172, 41346], [25905, 27216, 41151],
    [28902, 30156, 41280], [27513, 27339, 34680], [32775, 31824, 44433], [24690, 27147, 38580],
    [23439, 24477, 35154], [32697, 34047, 46527], [33399, 35013, 47556], [26889, 26895, 35295],
    [23013, 26046, 38283], [22626, 25062, 37224],
], dtype=float)
rp, rh, rr = RACK[:, 0], RACK[:, 1], RACK[:, 2]
labels = ["PPO", "Heuristic", "Random"]
means = [rp.mean(), rh.mean(), rr.mean()]
stds = [rp.std(), rh.std(), rr.std()]
cols2 = [COL["PPO"], COL["Heuristic"], COL["Random"]]

fig, ax = plt.subplots(figsize=(6.4, 5.2))
fig.suptitle("Raf Senaryosu — Tek Kat, Rota Optimizasyonu", fontsize=15, fontweight="bold")
bars = ax.bar(labels, means, yerr=stds, capsize=6, color=cols2, edgecolor="black", linewidth=0.6)
label_bars(ax, bars, means, "{:.0f}")
ax.set_ylabel("Vinç mesafesi (m, ort ± std)", fontsize=13)
ax.set_title("30 görülmemiş senaryo · düşük = iyi", fontsize=12)
ax.tick_params(labelsize=12)
wins = int((rp < rh).sum())
ax.text(0.5, -0.16, f"PPO, Heuristic'i {wins}/30 senaryoda yendi · "
        f"%+{100*(rh.mean()-rp.mean())/rh.mean():.1f} · Wilcoxon p<0.001",
        transform=ax.transAxes, ha="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf6ee", edgecolor=COL["PPO"]))
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(OUT / "fig2_raf_senaryo.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ============ FIG 3: METODOLOJİK DÜZELTME YOLCULUĞU (PPO rehandling) ============
stages = ["In-sample\n(ezber)", "Görülmemiş\n(ezber çöküşü)", "Havuz\n(affinity'siz)", "Affinity\nzorlanmış"]
vals = [3.40, 21.62, 4.13, 10.40]
scol = ["#cfcfcf", "#d9534f", "#f0ad4e", "#2e9e5b"]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
fig.suptitle("Metodolojik Titizlik — PPO Rehandling'inde İki Öz-Düzeltme",
             fontsize=15, fontweight="bold")
bars = ax.bar(stages, vals, color=scol, edgecolor="black", linewidth=0.6)
label_bars(ax, bars, vals, "{:.2f}")
ax.axhline(8.57, color="#3f7fc4", ls="--", lw=1.5)
ax.text(3.45, 8.57, " Heuristic = 8.57", color="#3f7fc4", va="bottom", ha="right", fontsize=11)
ax.set_ylabel("PPO rehandling (held-out ort)", fontsize=13)
ax.tick_params(labelsize=11)
ax.annotate("Düzeltme #1\noverfitting", xy=(1, 21.62), xytext=(1.1, 16),
            fontsize=10, ha="left", arrowprops=dict(arrowstyle="->", color="#d9534f"))
ax.annotate("Düzeltme #2\nkısıt-gevşemesi", xy=(3, 10.40), xytext=(2.3, 15),
            fontsize=10, ha="left", arrowprops=dict(arrowstyle="->", color="#2e9e5b"))
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig3_metodoloji.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("Kaydedildi ->", OUT.resolve())
for f in sorted(OUT.glob("*.png")):
    print("  ", f.name)
print("Font:", plt.rcParams["font.family"])
