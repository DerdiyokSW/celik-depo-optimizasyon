"""Senaryo havuzu üretici — train/test ayrımı için ayrık tohumlu veri setleri.

Çalıştırma: ``python -m src.data.generate_pool``

PPO overfitting'inin kök nedeni: ajan TEK bir sabit senaryoya (aynı bobin/sipariş
popülasyonu) eğitildi ve eval AYNI veriyi kullandı → genelleme değil ezber ölçüldü.
Çözüm: bir senaryo DAĞILIMI (havuz) üret; eğitimi train havuzunda yap, değerlendirmeyi
AYRIK (held-out, görülmemiş) test havuzunda yap.

Tohum aralıkları BİLİNÇLİ ayrıktır (sızıntı yok):
  - TRAIN: seed 0..63   (64 farklı popülasyon — eğitim havuzu)
  - TEST : seed 9000..9029 (30 görülmemiş popülasyon — held-out test havuzu)

Depo layout'u (8×36×2) HEPSİNDE sabittir; yalnızca bobin/sipariş/araç/başlangıç-
yerleşimi tohuma göre değişir. Böylece gözlem/aksiyon uzayı tüm senaryolarda aynıdır
ve genelleme iyi tanımlıdır (ağ boyutu değişmez).
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import GeneratorConfig
from .generate_all import build_dataset, write_dataset

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
POOL_DIR: Path = PROJECT_ROOT / "data" / "pool"

# Ayrık tohum aralıkları — train ile test ASLA örtüşmez (sızıntı önleme).
TRAIN_SEEDS: range = range(0, 64)        # 64 eğitim popülasyonu
TEST_SEEDS: range = range(9000, 9030)    # 30 görülmemiş test popülasyonu


def _generate_split(seeds: range, split_dir: Path, label: str) -> int:
    """Bir tohum aralığı için her tohumda bir veri seti üretip ``split_dir/seed_S``'e yazar."""
    split_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for s in seeds:
        out_dir = split_dir / f"seed_{s}"
        coils, vehicles, orders, layout, initial_state = build_dataset(GeneratorConfig(seed=s))
        write_dataset(coils, vehicles, orders, layout, initial_state, out_dir)
        count += 1
        print(f"  [{label}] seed={s:<5} -> {out_dir.relative_to(PROJECT_ROOT)} "
              f"({len(coils)} bobin, {len(orders)} sipariş)")
    return count


def generate_pool() -> None:
    """Train (0..63) ve held-out test (9000..9029) havuzlarını üretir."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Senaryo havuzu üretiliyor -> {POOL_DIR}")
    print(f"TRAIN tohumları: {TRAIN_SEEDS.start}..{TRAIN_SEEDS.stop - 1} "
          f"({len(TRAIN_SEEDS)} popülasyon)")
    n_train = _generate_split(TRAIN_SEEDS, POOL_DIR / "train", "train")
    print(f"TEST tohumları: {TEST_SEEDS.start}..{TEST_SEEDS.stop - 1} "
          f"({len(TEST_SEEDS)} popülasyon) — held-out, eğitimde GÖRÜLMEZ")
    n_test = _generate_split(TEST_SEEDS, POOL_DIR / "test", "test")

    print(f"\nHavuz hazır: train={n_train}, test={n_test} veri seti.")
    print(f"  train -> {(POOL_DIR / 'train').relative_to(PROJECT_ROOT)}")
    print(f"  test  -> {(POOL_DIR / 'test').relative_to(PROJECT_ROOT)}")
    # Sızıntı güvencesi: aralıkların kesişimi boş olmalı.
    assert set(TRAIN_SEEDS).isdisjoint(set(TEST_SEEDS)), "Train/test tohumları örtüşüyor!"
    print("Sızıntı yok: train ∩ test = ∅.")


if __name__ == "__main__":
    generate_pool()
