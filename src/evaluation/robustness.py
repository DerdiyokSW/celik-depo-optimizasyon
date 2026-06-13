"""Dayanıklılık analizi — olay hızını değiştirerek stres altında karşılaştırma.

Tek bir hızda kıyaslama yetmez; olay yoğunluğu (8/12/20 olay/saat) değiştirilerek
politikaların stres altındaki davranışı ölçülür. Beklenti: düşük yükte sezgisel
yeterli; yük arttıkça öğrenen/akıllı politikaların farkı açması. Çıktı: olay
hızına karşı rehandling eğrileri (runs/evaluation/robustness.png) + JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# matplotlib lazily imported inside _write_outputs to avoid an init segfault when
# this module is loaded after compare (which also configures matplotlib).
from src.ml.delay_model import DelayPredictor
from src.simulation.loaders import Scenario, load_pool

from .compare import DELAY_MODEL_PATH, OUT_DIR, TEST_POOL_DIR, build_policies
from .runner import evaluate_policy, evaluate_policy_on_pool
from .scenario import make_scenarios

DEFAULT_RATES = (8.0, 12.0, 20.0)


def robustness(
    n_scenarios: int = 30,
    base_seed: int = 2000,
    rates: tuple[float, ...] = DEFAULT_RATES,
    horizon_hours: float = 24.0,
) -> dict:
    """Her olay hızında politikaları kıyaslar; rehandling eğrilerini üretir/yazar."""
    scenario = Scenario.from_data_dir()
    delay_model = DelayPredictor.load(str(DELAY_MODEL_PATH)) if DELAY_MODEL_PATH.exists() else None
    policies = build_policies(delay_model)

    # curve[politika][hız] = ortalama rehandling
    curve: dict[str, dict[float, float]] = {name: {} for name in policies}
    for rate in rates:
        specs = make_scenarios(n_scenarios, base_seed, rate, horizon_hours)
        for name, policy in policies.items():
            df = evaluate_policy(policy, scenario, specs)
            curve[name][rate] = float(df["rehandling"].mean())
        print(f"Olay hızı {rate}/saat tamamlandı: "
              + ", ".join(f"{n}={curve[n][rate]:.1f}" for n in policies))

    _write_outputs(curve, list(rates), n_scenarios)
    return curve


def robustness_heldout(
    pool_dir=TEST_POOL_DIR,
    base_seed: int = 2000,
    rates: tuple[float, ...] = DEFAULT_RATES,
    horizon_hours: float = 24.0,
) -> dict:
    """HELD-OUT dayanıklılık: her olay hızında, GÖRÜLMEMİŞ test havuzu üzerinde kıyas.

    ``robustness``'tan farkı: tek senaryo yerine test havuzundaki tüm popülasyonları
    dolaşır (genelleşme stres testi). Her hızda her politika tüm popülasyonlarda koşulur,
    ortalama rehandling eğrisi üretilir.
    """
    scenarios = load_pool(pool_dir)
    delay_model = DelayPredictor.load(str(DELAY_MODEL_PATH)) if DELAY_MODEL_PATH.exists() else None
    policies = build_policies(delay_model)

    curve: dict[str, dict[float, float]] = {name: {} for name in policies}
    for rate in rates:
        for name, policy in policies.items():
            df = evaluate_policy_on_pool(policy, scenarios, base_seed, rate, horizon_hours)
            curve[name][rate] = float(df["rehandling"].mean())
        print(f"[held-out] Olay hızı {rate}/saat tamamlandı: "
              + ", ".join(f"{n}={curve[n][rate]:.1f}" for n in policies))

    _write_outputs(curve, list(rates), len(scenarios))
    return curve


def _write_outputs(curve: dict, rates: list[float], n_scenarios: int) -> None:
    """JSON + olay hızı/rehandling eğri grafiği yazar."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "robustness.json").write_text(
        json.dumps({"n_scenarios": n_scenarios, "curve": curve}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Lazy import: matplotlib'i yalnızca burada yükle (modül seviyesinde init segfault
    # gözlendi — compare zaten matplotlib'i ayarlıyor, ikinci yükleme çakışıyor).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, by_rate in curve.items():
        ax.plot(rates, [by_rate[r] for r in rates], marker="o", label=name)
    ax.set_xlabel("Olay hızı (olay/saat)")
    ax.set_ylabel("Ortalama rehandling")
    ax.set_title(f"Dayanıklılık — {n_scenarios} senaryo/hız")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "robustness.png", dpi=120)
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    parser = argparse.ArgumentParser(description="Dayanıklılık analizi")
    parser.add_argument("--in-sample", action="store_true",
                        help="data/ tek senaryosunda (eski); varsayılan held-out havuz")
    args = parser.parse_args()
    if args.in_sample:
        robustness()
    else:
        print(">>> HELD-OUT dayanıklılık (görülmemiş test havuzu).")
        robustness_heldout()
    print(f"\nDayanıklılık çıktıları yazıldı -> {OUT_DIR}")


if __name__ == "__main__":
    main()
