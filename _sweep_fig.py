"""GEÇİCİ — Yük/doluluk taraması poster figürü (TNR, sweep.json'dan). Bitince silinebilir."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(".").resolve()
data = json.loads((ROOT / "runs" / "evaluation" / "sweep.json").read_text())
rates = [float(r) for r in sorted(data.keys(), key=float)]
fills = [int(data[str(r)]["PPO"]["fill"] * 100) for r in rates]
xs = [f"{int(r)}/saat\n(%{f})" for r, f in zip(rates, fills)]
order = ["PPO", "Heuristic", "MLHeuristic", "Random"]
COL = {"Random": "#9aa0a6", "Heuristic": "#3f7fc4", "MLHeuristic": "#8e6fc9", "PPO": "#2e9e5b"}
MK = {"PPO": "o", "Heuristic": "s", "MLHeuristic": "^", "Random": "D"}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
fig.suptitle("Yük/Doluluk Taraması — Ana Senaryo (2 kat + affinity) · 30 görülmemiş senaryo",
             fontsize=15, fontweight="bold")

for met, ax, ylab in [("reh", a1, "Rehandling (ort)"), ("cra", a2, "Vinç mesafesi (m, ort)")]:
    for pol in order:
        ys = [data[str(r)][pol][met] for r in rates]
        ax.plot(range(len(rates)), ys, marker=MK[pol], color=COL[pol], lw=2.2, ms=9, label=pol)
        for i, y in enumerate(ys):
            ax.annotate(f"{y:.1f}" if met == "reh" else f"{y:.0f}",
                        (i, y), textcoords="offset points", xytext=(0, 7),
                        fontsize=8.5, ha="center", color=COL[pol])
    ax.set_xticks(range(len(rates))); ax.set_xticklabels(xs, fontsize=10)
    ax.set_ylabel(ylab, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.25)

a1.set_title("Rehandling — düşük = iyi", fontsize=12)
a1.legend(fontsize=10, loc="upper left")
n = len(rates)
ppo_reh = [data[str(r)]["PPO"]["reh"] for r in rates]
a1.annotate(f"Düşük dolulukta\nPPO en az rehandling\n({ppo_reh[0]:.1f})", xy=(0, ppo_reh[0]),
            xytext=(0.4, max(ppo_reh) * 0.42), fontsize=9.5, color="#2e9e5b",
            arrowprops=dict(arrowstyle="->", color="#2e9e5b"))
a1.annotate("Yüksek dolulukta\nPPO hızla artıyor", xy=(n - 1, ppo_reh[-1]),
            xytext=(n - 2.4, ppo_reh[-1] * 0.78), fontsize=9.5, color="#d9534f",
            arrowprops=dict(arrowstyle="->", color="#d9534f"))
a2.set_title("Vinç mesafesi — düşük = iyi (en düşük: Heuristic)", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out = ROOT / "runs" / "evaluation" / "poster" / "fig4_yuk_taramasi.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("Kaydedildi ->", out)
