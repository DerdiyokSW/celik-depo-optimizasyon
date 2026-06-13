"""Dört politikayı tohumlanmış senaryolarda karşılaştırır + istatistiksel test (main).

Çalıştırma: ``python -m src.evaluation.compare``

Aynı senaryolar tüm politikalara verilir (eşleştirilmiş karşılaştırma); her metrik
için ortalama ± standart sapma (sapma = kararlılık ölçüsü) ve rehandling üzerinde
Wilcoxon signed-rank p-değerleri raporlanır. Çıktı: konsol tablosu + JSON + CSV +
bar grafiği (runs/evaluation/). Tüm sayılar gerçek koşum çıktısıdır.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")  # başsız (headless) çizim
import matplotlib.pyplot as plt  # noqa: E402

from src.ml.delay_model import DelayPredictor
from src.policies import HeuristicPolicy, MLHeuristicPolicy, PPOPolicy, RandomPolicy
from src.simulation.loaders import Scenario, load_pool

from .runner import evaluate_policy, evaluate_policy_on_pool
from .scenario import make_scenarios

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DELAY_MODEL_PATH: Path = PROJECT_ROOT / "models" / "delay_model.txt"
# Held-out (görülmemiş) test havuzu — genelleşme/overfitting ölçümünün temeli.
TEST_POOL_DIR: Path = PROJECT_ROOT / "data" / "pool" / "test"
# Değerlendirme için tercih edilen PPO modeli: eval-callback'in seçtiği EN İYİ;
# yoksa son kaydedilen. (Eval-best, eğitim sonu modeline kıyasla genelde daha iyidir.)
_PPO_BEST_PATH: Path = PROJECT_ROOT / "models" / "ppo_best" / "best_model.zip"
_PPO_LAST_PATH: Path = PROJECT_ROOT / "models" / "ppo_warehouse.zip"
PPO_MODEL_PATH: Path = _PPO_BEST_PATH if _PPO_BEST_PATH.exists() else _PPO_LAST_PATH
OUT_DIR: Path = PROJECT_ROOT / "runs" / "evaluation"

# Tabloda gösterilen metrikler (düşük = iyi olanlar; doluluk bilgi amaçlı).
METRICS = ["rehandling", "crane_distance_m", "loading_time_min", "decision_ms", "fill_ratio"]


def build_policies(delay_model: DelayPredictor | None) -> dict:
    """Mevcut politikaları kurar. ML yalnızca gecikme modeli, PPO yalnızca eğitilmiş
    model dosyası varsa eklenir."""
    policies: dict = {"Random": RandomPolicy(seed=0), "Heuristic": HeuristicPolicy()}
    if delay_model is not None:
        policies["MLHeuristic"] = MLHeuristicPolicy(delay_model)
    if PPO_MODEL_PATH.exists():
        policies["PPO"] = PPOPolicy(str(PPO_MODEL_PATH), delay_model=delay_model)
    return policies


def compare(
    n_scenarios: int = 50,
    base_seed: int = 1000,
    event_rate_per_hour: float = 12.0,
    horizon_hours: float = 24.0,
) -> dict:
    """Politikaları N senaryoda kıyaslar; özet + Wilcoxon döndürür ve diske yazar."""
    scenario = Scenario.from_data_dir()
    delay_model = DelayPredictor.load(str(DELAY_MODEL_PATH)) if DELAY_MODEL_PATH.exists() else None
    specs = make_scenarios(n_scenarios, base_seed, event_rate_per_hour, horizon_hours)
    policies = build_policies(delay_model)

    # Her politikayı aynı senaryolarda koştur.
    results = {name: evaluate_policy(policy, scenario, specs) for name, policy in policies.items()}

    # Özet: metrik başına ortalama ± std.
    summary = {
        name: {met: {"mean": float(df[met].mean()), "std": float(df[met].std())} for met in METRICS}
        for name, df in results.items()
    }

    # Wilcoxon (eşleştirilmiş) — rehandling üzerinde her politika Random'a karşı.
    stats = _wilcoxon_vs_baseline(results, baseline="Random", metric="rehandling")
    # Ek çiftler: Heuristic vs ML, ML vs PPO (varsa).
    stats.update(_wilcoxon_pairs(results, [("Heuristic", "MLHeuristic"), ("MLHeuristic", "PPO")], "rehandling"))

    report = {
        "n_scenarios": n_scenarios, "base_seed": base_seed,
        "event_rate_per_hour": event_rate_per_hour, "horizon_hours": horizon_hours,
        "summary": summary, "wilcoxon_rehandling": stats,
        "ppo_included": "PPO" in policies,
    }
    _write_outputs(results, report)
    _print_table(summary, stats)
    return report


def compare_heldout(
    pool_dir: Path = TEST_POOL_DIR,
    base_seed: int = 2000,
    event_rate_per_hour: float = 12.0,
    horizon_hours: float = 24.0,
) -> dict:
    """HELD-OUT (görülmemiş) test havuzunda dört politikayı kıyaslar — DÜRÜST genelleşme.

    ``compare``'den farkı: tek bir (eğitimde görülen) senaryo yerine, eğitim tohumlarıyla
    ÖRTÜŞMEYEN test havuzundaki HER popülasyonu dolaşır. Her popülasyona deterministik bir
    olay tohumu verilir; dört politika da aynı popülasyon+tohumu görür (eşleştirilmiş).
    Bu, PPO'nun ezber yerine GENELLEŞTİRME yeteneğini ölçer (overfitting düzeltmesinin
    başarı kriteri). Çıktı şeması ``compare`` ile aynıdır; rapor ``held_out=True`` taşır.
    """
    scenarios = load_pool(pool_dir)
    delay_model = DelayPredictor.load(str(DELAY_MODEL_PATH)) if DELAY_MODEL_PATH.exists() else None
    policies = build_policies(delay_model)

    # Her politikayı test havuzundaki tüm popülasyonlarda koştur (popülasyon başına 1 satır).
    results = {
        name: evaluate_policy_on_pool(policy, scenarios, base_seed, event_rate_per_hour, horizon_hours)
        for name, policy in policies.items()
    }

    summary = {
        name: {met: {"mean": float(df[met].mean()), "std": float(df[met].std())} for met in METRICS}
        for name, df in results.items()
    }
    stats = _wilcoxon_vs_baseline(results, baseline="Random", metric="rehandling")
    stats.update(_wilcoxon_pairs(results, [("Heuristic", "MLHeuristic"), ("MLHeuristic", "PPO")], "rehandling"))

    report = {
        "held_out": True, "n_populations": len(scenarios), "base_seed": base_seed,
        "event_rate_per_hour": event_rate_per_hour, "horizon_hours": horizon_hours,
        "summary": summary, "wilcoxon_rehandling": stats,
        "ppo_included": "PPO" in policies,
    }
    _write_outputs(results, report)
    _print_table(summary, stats)
    return report


def _wilcoxon_vs_baseline(results: dict, baseline: str, metric: str) -> dict:
    """Her politikanın metriğini baseline ile eşleştirilmiş Wilcoxon ile kıyaslar."""
    out: dict = {}
    base = results[baseline][metric].to_numpy()
    for name, df in results.items():
        if name == baseline:
            continue
        out[f"{name}_vs_{baseline}"] = _safe_wilcoxon(base, df[metric].to_numpy())
    return out


def _wilcoxon_pairs(results: dict, pairs: list[tuple[str, str]], metric: str) -> dict:
    """Belirli politika çiftleri için eşleştirilmiş Wilcoxon."""
    out: dict = {}
    for a, b in pairs:
        if a in results and b in results:
            out[f"{a}_vs_{b}"] = _safe_wilcoxon(results[a][metric].to_numpy(), results[b][metric].to_numpy())
    return out


def _safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon'u güvenli koşar (tüm farklar sıfırsa test tanımsızdır)."""
    diff = a - b
    if np.all(diff == 0):
        return {"p_value": 1.0, "note": "tüm farklar sıfır (özdeş)"}
    try:
        stat, p = wilcoxon(a, b)
        return {"statistic": float(stat), "p_value": float(p),
                "mean_diff": float(np.mean(diff))}
    except ValueError as exc:
        return {"p_value": None, "note": str(exc)}


def _write_outputs(results: dict, report: dict) -> None:
    """JSON raporu, politika başına CSV ve rehandling bar grafiğini yazar."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for name, df in results.items():
        df.to_csv(OUT_DIR / f"results_{name}.csv", index=False)

    # Rehandling ortalama ± std bar grafiği.
    names = list(results.keys())
    means = [results[n]["rehandling"].mean() for n in names]
    stds = [results[n]["rehandling"].std() for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, means, yerr=stds, capsize=5, color=["#888", "#3498db", "#9b59b6", "#e67e22"][: len(names)])
    ax.set_ylabel("Rehandling (ortalama ± std)")
    # Held-out raporda 'n_populations', in-sample raporda 'n_scenarios' bulunur.
    if report.get("held_out"):
        ax.set_title(f"Held-out Kıyas — {report['n_populations']} görülmemiş popülasyon")
    else:
        ax.set_title(f"Politika Karşılaştırması — {report['n_scenarios']} senaryo")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rehandling_bar.png", dpi=120)
    plt.close(fig)


def _print_table(summary: dict, stats: dict) -> None:
    """Konsola özet tablo + Wilcoxon p-değerleri yazar."""
    print("\n=== Politika Karşılaştırması (ortalama ± std) ===")
    header = f"{'Politika':<13}" + "".join(f"{m:>20}" for m in METRICS)
    print(header)
    for name, mets in summary.items():
        row = f"{name:<13}"
        for m in METRICS:
            row += f"{mets[m]['mean']:>10.2f}±{mets[m]['std']:<8.2f}"
        print(row)
    print("\n=== Wilcoxon (rehandling, eşleştirilmiş) p-değerleri ===")
    for pair, res in stats.items():
        p = res.get("p_value")
        md = res.get("mean_diff")
        extra = f"  ort_fark={md:+.2f}" if md is not None else f"  ({res.get('note','')})"
        print(f"  {pair:<28}: p={p}{extra}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Varsayılan: HELD-OUT değerlendirme (görülmemiş test havuzu) — dürüst genelleşme.
    # In-sample (tek data/ senaryosu) kıyas için: --in-sample.
    import argparse
    parser = argparse.ArgumentParser(description="Politika karşılaştırması")
    parser.add_argument("--in-sample", action="store_true",
                        help="data/ tek senaryosunda kıyasla (eski; ezber riski)")
    args = parser.parse_args()

    if args.in_sample:
        print(">>> IN-SAMPLE kıyas (tek senaryo, data/).")
        report = compare()
    else:
        print(">>> HELD-OUT kıyas (görülmemiş test havuzu, data/pool/test).")
        report = compare_heldout()
    print(f"\nÇıktılar yazıldı -> {OUT_DIR}")
    print("PPO dahil mi:", report["ppo_included"])


if __name__ == "__main__":
    main()
