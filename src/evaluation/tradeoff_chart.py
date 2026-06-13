"""Çok-amaçlı tradeoff grafiği (C2) — rehandling / vinç mesafesi / yükleme süresi.

``python -m src.evaluation.tradeoff_chart``

``compare`` çıktısındaki ``comparison.json``'dan üç metriğin politika ortalamalarını
okur, her metriği kendi içinde [0,1]'e normalize eder (en kötü = 1.0) ve gruplanmış
bir bar grafiği üretir. Amaç: rehandling ile toplam vinç mesafesinin KISMEN ÇELİŞEN
hedefler olduğunu görsel kılmak (problemin çok-amaçlı doğası). Grafiğin altına,
gerçek sayılardan türetilen kısa bir yorum bloğu yazılır (uydurma sayı yok).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # başsız çizim
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
OUT_DIR: Path = PROJECT_ROOT / "runs" / "evaluation"
COMPARISON_JSON: Path = OUT_DIR / "comparison.json"

# Gösterilecek üç maliyet metriği (hepsinde düşük = iyi). Etiketler grafiğe yazılır.
TRADEOFF_METRICS: list[tuple[str, str]] = [
    ("rehandling", "Rehandling"),
    ("crane_distance_m", "Vinç mesafesi"),
    ("loading_time_min", "Yükleme süresi"),
]


def _load_summary() -> dict:
    """comparison.json'dan politika→metrik→{mean,std} özetini okur."""
    if not COMPARISON_JSON.exists():
        raise FileNotFoundError(
            f"{COMPARISON_JSON} yok. Önce 'python -m src.evaluation.compare' koş."
        )
    report = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))
    return report["summary"]


def _normalize(means: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Her metriği politikalar arası en büyük değere bölerek [0,1]'e indirir.

    Her metrik kendi ölçeğinde (rehandling ~birim, mesafe ~on binler) olduğundan
    aynı eksende kıyaslanamaz; normalize edince en kötü politika 1.0, diğerleri
    oransal olur. Düşük = iyi yorumu korunur.
    """
    norm: dict[str, dict[str, float]] = {p: {} for p in means}
    for metric, _label in TRADEOFF_METRICS:
        peak = max(means[p][metric] for p in means) or 1.0
        for p in means:
            norm[p][metric] = means[p][metric] / peak
    return norm


def _best_policy(means: dict, metric: str) -> tuple[str, float]:
    """Bir metrikte en düşük (en iyi) ortalamaya sahip politikayı döndürür."""
    best = min(means, key=lambda p: means[p][metric])
    return best, means[best][metric]


def build_tradeoff_chart() -> dict:
    """Normalize gruplanmış bar grafiği + veri-güdümlü yorum üretir; diske yazar."""
    summary = _load_summary()
    policies = list(summary.keys())
    means = {p: {m: summary[p][m]["mean"] for m, _ in TRADEOFF_METRICS} for p in policies}
    norm = _normalize(means)

    # Gruplanmış bar: x ekseni = 3 metrik, her grupta politika barları.
    n_metrics = len(TRADEOFF_METRICS)
    n_pol = len(policies)
    bar_w = 0.8 / n_pol
    x = range(n_metrics)
    colors = ["#888", "#3498db", "#9b59b6", "#e67e22"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, p in enumerate(policies):
        offsets = [xi + (i - (n_pol - 1) / 2) * bar_w for xi in x]
        heights = [norm[p][m] for m, _ in TRADEOFF_METRICS]
        ax.bar(offsets, heights, width=bar_w, label=p, color=colors[i % len(colors)])

    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in TRADEOFF_METRICS])
    ax.set_ylabel("Normalize maliyet (en kötü = 1.0; düşük = iyi)")
    ax.set_title("Çok-Amaçlı Tradeoff — Politika × Metrik")
    ax.legend()
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "tradeoff_bar.png", dpi=120)
    plt.close(fig)

    # Veri-güdümlü yorum (gerçek sayılardan; uydurma yok).
    best_reh, reh_val = _best_policy(means, "rehandling")
    best_crane, crane_val = _best_policy(means, "crane_distance_m")
    comment = (
        f"**Çok-amaçlı tradeoff yorumu** (kaynak: comparison.json, gerçek koşum):\n\n"
        f"- En düşük **rehandling**: **{best_reh}** ({reh_val:.2f}) — pahalı sevkiyat "
        f"operasyonunu (üst bobini kaldırma) minimize eder.\n"
        f"- En düşük **vinç mesafesi**: **{best_crane}** ({crane_val:.0f} m) — toplam "
        f"vinç hareketini (mesafe-açgözlü) minimize eder.\n\n"
        f"Bu iki hedefin farklı politikalarda minimize olması, problemin **çok-amaçlı** "
        f"doğasını gösterir: rehandling ve toplam vinç mesafesi kısmen çelişen "
        f"hedeflerdir. {best_reh}, az-rehandling için bazen daha uzak ama 'temiz' "
        f"(engelsiz) konumlar seçerek mesafeyi artırır; {best_crane} ise her hamlede en "
        f"yakın konumu seçtiğinden mesafeyi düşürür ama acil bobinleri bazen gömerek "
        f"rehandling'i artırır. Operasyonel öncelik (vinç enerjisi mi, sevkiyat hızı mı) "
        f"hangi politikanın tercih edileceğini belirler.\n"
    )
    (OUT_DIR / "tradeoff_comment.md").write_text(comment, encoding="utf-8")
    return {"means": means, "best_rehandling": best_reh, "best_crane": best_crane}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = build_tradeoff_chart()
    print("Tradeoff grafiği yazıldı -> runs/evaluation/tradeoff_bar.png")
    print("Yorum yazıldı       -> runs/evaluation/tradeoff_comment.md")
    print(f"En düşük rehandling : {result['best_rehandling']}")
    print(f"En düşük vinç mesafe: {result['best_crane']}")


if __name__ == "__main__":
    main()
