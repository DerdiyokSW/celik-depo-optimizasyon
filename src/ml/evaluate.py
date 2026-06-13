"""Gecikme modeli değerlendirme metrikleri ve rapor yazımı.

Metrikler test setinde hesaplanır (docs/04 §7). Önemli: veri üretici gecikmeye
std ~8 dk indirgenemez Gaussian gürültü ekler; bu, RMSE için ~8 dk teorik alt
sınır demektir. R²'nin 1.0 olmaması bir kusur değil, gerçekçiliğin sonucudur.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tahminin "isabetli" sayıldığı hata eşiği (dakika).
TOLERANCE_MIN: float = 15.0


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, tolerance_min: float = TOLERANCE_MIN
) -> dict[str, float]:
    """Tahmin metriklerini hesaplar: MAE, RMSE, R² ve ≤eşik isabet oranı.

    Dönüş: {'mae', 'rmse', 'r2', 'within_tolerance_ratio'} (hepsi float).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    # Tahminlerin yüzde kaçı eşik dakika içinde isabetli.
    within = float(np.mean(np.abs(y_true - y_pred) <= tolerance_min))
    return {"mae": mae, "rmse": rmse, "r2": r2, "within_tolerance_ratio": within}


def write_report(report: dict, path: str) -> None:
    """Metrik raporunu JSON olarak yazar (dizin yoksa oluşturur)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
