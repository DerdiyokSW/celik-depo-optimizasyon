"""Aşama 3 giriş noktası: model karşılaştırma, hiperparametre ayarı, final eğitim.

Çalıştırma: ``python -m src.ml.train``

Tek bir modele körü körüne gidilmez (savunmada "neden bu model" sorusuna cevap):
  1. Dört model 5-katlı CV ile karşılaştırılır (Lineer, Karar Ağacı, Random Forest,
     LightGBM) — ortalama MAE'ye göre.
  2. Kazanan (beklenen: LightGBM) için küçük bir ızgara üzerinden CV ile temel
     hiperparametreler ayarlanır.
  3. Final model tüm eğitim verisiyle eğitilip ayrılan test setinde ölçülür,
     ``models/delay_model.txt``e kaydedilir; rapor ``runs/delay_model_report.json``a yazılır.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

from .delay_model import DEFAULT_PARAMS, DelayPredictor
from .evaluate import compute_metrics, write_report
from .features import build_feature_matrix

# Proje kökü: bu dosya src/ml/ altında olduğundan iki üst dizin köktür.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_PATH: Path = PROJECT_ROOT / "data" / "vehicles_12m.parquet"
MODEL_PATH: Path = PROJECT_ROOT / "models" / "delay_model.txt"
REPORT_PATH: Path = PROJECT_ROOT / "runs" / "delay_model_report.json"

RANDOM_STATE: int = 42
CV_FOLDS: int = 5


def compare_models(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Dört modeli 5-katlı CV ile karşılaştırır; her birinin ortalama MAE'sini döndürür."""
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(**DEFAULT_PARAMS, random_state=RANDOM_STATE, verbose=-1),
    }
    results: dict[str, float] = {}
    for name, model in candidates.items():
        # neg MAE skorlarının ortalamasının negatifi = ortalama MAE.
        scores = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=kf)
        results[name] = float(-scores.mean())
    return results


def tune_lightgbm(X: pd.DataFrame, y: pd.Series) -> dict:
    """LightGBM için küçük bir ızgara üzerinde CV ile en iyi hiperparametreleri seçer.

    Sabit parametreler (subsample vb.) korunur; yalnızca temel olanlar aranır.
    Dönüş: DEFAULT_PARAMS ile birleştirilmiş en iyi parametre sözlüğü.
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    base = lgb.LGBMRegressor(
        random_state=RANDOM_STATE,
        verbose=-1,
        subsample=DEFAULT_PARAMS["subsample"],
        colsample_bytree=DEFAULT_PARAMS["colsample_bytree"],
        min_child_samples=DEFAULT_PARAMS["min_child_samples"],
    )
    grid = {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
    }
    search = GridSearchCV(base, grid, scoring="neg_mean_absolute_error", cv=kf, n_jobs=-1)
    search.fit(X, y)
    best = dict(DEFAULT_PARAMS)
    best.update(search.best_params_)
    return best


def main() -> None:
    """Karşılaştırma + ayar + final eğitim hattını çalıştırır."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    vehicles = pd.read_parquet(DATA_PATH)
    X, y = build_feature_matrix(vehicles)
    print(f"Veri yüklendi: {len(X)} araç, {X.shape[1]} özellik.\n")

    # 1) Model karşılaştırma.
    print("5-katlı CV ile model karşılaştırması (ortalama MAE, dakika):")
    cv_results = compare_models(X, y)
    for name, mae in sorted(cv_results.items(), key=lambda kv: kv[1]):
        print(f"  {name:<18}: {mae:.3f}")
    winner = min(cv_results, key=cv_results.get)
    print(f"\nCV kazananı: {winner}")

    # 2) LightGBM hiperparametre ayarı (proje yıldızı ve teslim edilen model LightGBM).
    print("\nLightGBM hiperparametreleri ayarlanıyor (ızgara + CV)...")
    best_params = tune_lightgbm(X, y)
    print(f"  En iyi parametreler: {best_params['n_estimators']} ağaç, "
          f"lr={best_params['learning_rate']}, num_leaves={best_params['num_leaves']}")

    # 3) Final eğitim + değerlendirme.
    predictor = DelayPredictor(params=best_params, random_state=RANDOM_STATE)
    predictor.train(vehicles)
    y_pred = predictor.predict_from_features(predictor._X_test)
    metrics = compute_metrics(predictor._y_test.values, y_pred)
    print("\nTest seti metrikleri:")
    print(f"  MAE  = {metrics['mae']:.2f} dk")
    print(f"  RMSE = {metrics['rmse']:.2f} dk  (üretici gürültüsü ~8 dk alt sınır)")
    print(f"  R²   = {metrics['r2']:.3f}")
    print(f"  ≤15 dk isabet oranı = {metrics['within_tolerance_ratio']:.2%}")

    # Modeli kaydet.
    predictor.save(str(MODEL_PATH))
    print(f"\nModel kaydedildi -> {MODEL_PATH}")

    # Raporu yaz.
    report = {
        "cv_comparison_mae": cv_results,
        "chosen_model": winner,
        "deployed_model": "LightGBM",
        "best_params": best_params,
        "test_metrics": metrics,
        "feature_importances": predictor.feature_importances(),
        "n_total": int(len(X)),
        "random_state": RANDOM_STATE,
    }
    write_report(report, str(REPORT_PATH))
    print(f"Rapor yazıldı -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
