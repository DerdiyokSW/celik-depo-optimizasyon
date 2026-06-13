"""GEÇİCİ — Eksik şemaları (mimari, DFD, geometri) ve denklem PNG'lerini üretir.

Çıktı:
  runs/evaluation/poster/fig_mimari.png    (Şekil 2.1)
  runs/evaluation/poster/fig_dfd.png       (Şekil 2.2)
  runs/evaluation/poster/fig_geometri.png  (Şekil 3.1)
  runs/evaluation/poster/denklemler/eq_3_*.png  (Denklem 3.1–3.10)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("runs/evaluation/poster")
EQ = OUT / "denklemler"
EQ.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, GREY = "#3f7fc4", "#e0922f", "#2e9e5b", "#555555"


def box(ax, x, y, w, h, text, fc="#eaf2fb", ec=BLUE, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p1, p2, label="", off=0.08, color=GREY):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, lw=1.4,
                                 color=color, shrinkA=2, shrinkB=2))
    if label:
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + off, label, ha="center",
                fontsize=8, color=color)


# ---------- Şekil 2.1 — Sistem Mimarisi ----------
fig, ax = plt.subplots(figsize=(9.5, 6)); ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
ax.set_title("Şekil 2.1. Sistem Mimarisi (dört katman + görselleştirme)", fontsize=13, fontweight="bold")
box(ax, 3.2, 6.7, 3.6, 0.9, "Görselleştirme — 3B Dijital İkiz\n(Plotly / Dash)", fc="#fdeccd", ec=ORANGE)
box(ax, 0.2, 4.0, 2.7, 1.2, "Veri Üretici\n~5000 bobin · 3600 araç\n1200 sipariş", fs=9)
box(ax, 3.6, 4.0, 2.8, 1.2, "Simülasyon Çekirdeği\n8×36×2 = 576 konum\nvinç + kısıtlar + olaylar", fs=9)
box(ax, 7.1, 4.0, 2.7, 1.2, "Yerleştirme Politikaları\nRandom · Heuristic\nML-Heuristic · PPO", fs=9)
box(ax, 3.6, 1.4, 2.8, 1.0, "Gecikme Tahmin ML\nLightGBM (MAE ≈ 6,95 dk)", fc="#e7f3ec", ec=GREEN, fs=9)
arrow(ax, (2.9, 4.6), (3.6, 4.6), "CSV / Parquet")
arrow(ax, (6.4, 4.75), (7.1, 4.75), "gözlem + geçerli konum")
arrow(ax, (7.1, 4.35), (6.4, 4.35), "seçilen slot")
arrow(ax, (4.6, 2.4), (4.6, 4.0), "gecikme")
arrow(ax, (6.4, 1.9), (8.0, 4.0), "aciliyet")
arrow(ax, (5.0, 5.2), (5.0, 6.7), "durum")
fig.savefig(OUT / "fig_mimari.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Şekil 2.2 — Veri Akış Diyagramı (DFD) ----------
fig, ax = plt.subplots(figsize=(9.5, 5.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
ax.set_title("Şekil 2.2. Veri Akış Diyagramı (DFD)", fontsize=13, fontweight="bold")
box(ax, 0.2, 4.6, 2.2, 1.0, "Veri Üretici", fs=10)
box(ax, 0.2, 2.4, 2.2, 1.0, "Olay Üreteci\n(Poisson)", fs=9)
box(ax, 3.2, 3.5, 2.4, 1.0, "Simülasyon\nÇekirdeği", fs=10)
box(ax, 6.2, 3.5, 2.4, 1.0, "Politika\n(Karar)", fs=10)
box(ax, 9.2, 3.5, 2.4, 1.0, "Yerleştirme /\nSevkiyat", fs=9)
box(ax, 6.2, 1.0, 2.4, 1.0, "Gecikme Tahmin\nML", fc="#e7f3ec", ec=GREEN, fs=9)
box(ax, 9.2, 1.0, 2.4, 1.0, "Metrikler\n(reh, vinç, doluluk)", fc="#f3eefb", ec="#8e6fc9", fs=8)
arrow(ax, (2.4, 5.1), (3.6, 4.5), "popülasyon")
arrow(ax, (2.4, 2.9), (3.8, 3.5), "olay akışı")
arrow(ax, (5.6, 4.0), (6.2, 4.0), "bekleyen bobin")
arrow(ax, (8.6, 4.0), (9.2, 4.0), "slot")
arrow(ax, (7.4, 2.0), (7.4, 3.5), "gecikme")
arrow(ax, (10.4, 3.5), (10.4, 2.0), "rehandling")
arrow(ax, (9.2, 3.7), (8.6, 3.7), "geri besleme", off=-0.25)
fig.savefig(OUT / "fig_dfd.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Şekil 3.1 — Depo Geometrisi ve Vinç Mesafesi ----------
fig, ax = plt.subplots(figsize=(9.5, 6)); ax.set_xlim(-1.5, 12); ax.set_ylim(-1.5, 9); ax.axis("off")
ax.set_title("Şekil 3.1. Depo Geometrisi ve Vinç Mesafe Modeli (üstten görünüm)",
             fontsize=13, fontweight="bold")
NB, NZ = 12, 8  # şematik (gerçek: 36 bay × 8 zone)
for z in range(NZ):
    for b in range(NB):
        ax.add_patch(Rectangle((b, z), 0.96, 0.96, fc="#eef3f8", ec="#c9d6e5", lw=0.6))
# Sevkiyat kapıları (sol kenar, bay 0)
for z in range(NZ):
    ax.add_patch(Rectangle((-0.9, z + 0.15), 0.6, 0.66, fc="#bfe3c8", ec=GREEN, lw=1.0))
ax.text(-1.4, NZ / 2, "Sevkiyat kapıları\n(bay 0)", rotation=90, va="center", ha="center", fontsize=9, color=GREEN)
# Üretim çıkışı (giriş)
ax.add_patch(Rectangle((NB + 0.2, -0.1), 1.2, 0.9, fc="#fdeccd", ec=ORANGE, lw=1.2))
ax.text(NB + 0.8, 0.35, "Üretim\nçıkışı", ha="center", va="center", fontsize=8, color=ORANGE)
# Köprü / araba eksenleri
ax.annotate("", xy=(NB + 0.1, -0.9), xytext=(0, -0.9), arrowprops=dict(arrowstyle="<->", color="#333"))
ax.text(NB / 2, -1.25, "Bay ekseni (araba) — kapıdan uzaklık", ha="center", fontsize=9)
ax.annotate("", xy=(-1.25, NZ), xytext=(-1.25, 0), arrowprops=dict(arrowstyle="<->", color="#333"))
ax.text(-1.25, NZ / 2 + 1.2, "Zone ekseni (köprü)", rotation=90, ha="center", fontsize=9)
# Chebyshev örneği: iki nokta
ax.plot([2.5], [1.5], "o", color=BLUE, ms=10)
ax.plot([8.5], [5.5], "o", color="#d9534f", ms=10)
ax.annotate("", xy=(8.5, 1.5), xytext=(2.5, 1.5), arrowprops=dict(arrowstyle="->", color=BLUE, ls="--"))
ax.annotate("", xy=(8.5, 5.5), xytext=(8.5, 1.5), arrowprops=dict(arrowstyle="->", color="#d9534f", ls="--"))
ax.text(5.5, 1.15, "|Δbay|", color=BLUE, fontsize=9, ha="center")
ax.text(8.75, 3.5, "|Δzone|", color="#d9534f", fontsize=9)
ax.text(6.0, 7.7, "Köprü ve araba EŞZAMANLI hareket eder →\nyatay mesafe = max(|Δbay|, |Δzone|)  (Chebyshev)",
        fontsize=10, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe6", ec="#e0c84a"))
ax.text(6.0, -1.55, "8 zone × 36 bay × 2 kat = 576 konum (tek katlı: 288)  ·  dikey bileşen: kat farkı",
        fontsize=9, ha="center", style="italic")
fig.savefig(OUT / "fig_geometri.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Denklem PNG'leri (mathtext) ----------
EQS = {
    "3_1": r"$d(a,b) = \max\,(\,|x_a - x_b|,\ |y_a - y_b|\,) + h_{kat}\,|\ell_a - \ell_b|$",
    "3_2": r"$c_{vinc}(slot) = d(g_{in},\ slot) + d(slot,\ g_{out})$",
    "3_3": r"$\rho = \frac{N_{dolu}}{N_{toplam}}$",
    "3_4": r"$r_t = -\,(\,w_c\,\hat{c}_{vinc,t} + w_r\,\Delta reh_t\,)$",
    "3_5": r"$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\,[\,\min(\,r_t(\theta)\,\hat{A}_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\,\hat{A}_t\,)\,]$",
    "3_6": r"$\hat{A}_t = \sum_{l\geq 0}(\gamma\lambda)^l\,\delta_{t+l},\qquad \delta_t = r_t + \gamma\,V(s_{t+1}) - V(s_t)$",
    "3_7": r"$L(\theta) = L^{CLIP}(\theta) - c_1\,L^{VF}(\theta) + c_2\,S[\pi_\theta](s_t)$",
    "3_8": r"$u(c) = \mathrm{clip}\,\left(1 - \frac{t_{kalan}}{H_a},\ 0,\ 1\right)$",
    "3_9": r"$a(slot) = 0.6\,a_{kat} + 0.4\,a_{cikis}$",
    "3_10": r"$\mathrm{uyum}(c,slot) = 1 - |\,u(c) - a(slot)\,|$",
}
try:
    plt.rcParams["mathtext.fontset"] = "stix"
except Exception:
    pass
for key, eq in EQS.items():
    fig = plt.figure(figsize=(8.2, 1.1))
    fig.text(0.02, 0.5, eq, fontsize=20, ha="left", va="center")
    fig.savefig(EQ / f"eq_{key}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

print("Şemalar -> fig_mimari/fig_dfd/fig_geometri.png")
print("Denklemler ->", len(EQS), "adet eq_3_*.png  ( ", EQ, ")")
print("DONE")
