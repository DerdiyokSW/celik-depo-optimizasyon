"""DelayPredictor — araç gecikmesini tahmin eden gözetimli model sarmalayıcısı.

Eğitim, tahmin ve kalıcılaştırmayı tek temiz arayüz altında toplar. Çekirdek
model LightGBM'dir (gradyan artırmalı ağaçlar); doğrusal olmayan etkileşimleri
(Aşama 1'deki mesafe×hava etkileşimi gibi) yakalamada güçlüdür. Aşama 4 ve 6
yalnızca ``predict`` / ``predict_batch`` kullanır.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.domain import Vehicle

from .features import FEATURE_COLUMNS, build_feature_matrix, vehicle_to_frame

# Varsayılan LightGBM hiperparametreleri. train.py bunları CV ile ayarlayıp
# kazanan kombinasyonu DelayPredictor'a verir; buradakiler makul başlangıçtır.
DEFAULT_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_samples": 20,
}


class DelayPredictor:
    """Araç gecikmesini (dakika) tahmin eden LightGBM tabanlı model sarmalayıcısı."""

    def __init__(self, params: dict | None = None, random_state: int = 42) -> None:
        self.params = dict(params) if params is not None else dict(DEFAULT_PARAMS)
        self.random_state = random_state
        self._booster: lgb.Booster | None = None
        self.feature_names: list[str] = list(FEATURE_COLUMNS)
        # Eğitimde ayrılan test seti (train.py değerlendirme için kullanır).
        self._X_test: pd.DataFrame | None = None
        self._y_test: pd.Series | None = None

    def train(self, vehicles: pd.DataFrame, test_size: float = 0.2) -> None:
        """Modeli eğitir: özellik matrisi üret, train/test böl, LightGBM eğit.

        Ayrılan test seti ``self._X_test/_y_test``e saklanır; sabit ``random_state``
        ile bölme ve eğitim tekrarlanabilirdir (determinizm).
        """
        X, y = build_feature_matrix(vehicles)
        if y is None:
            raise ValueError("Eğitim için 'delay_minutes' hedef sütunu gerekli.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        model = lgb.LGBMRegressor(**self.params, random_state=self.random_state, verbose=-1)
        model.fit(X_train, y_train)
        self._booster = model.booster_
        self._X_test, self._y_test = X_test, y_test

    def predict_batch(self, vehicles: pd.DataFrame) -> np.ndarray:
        """Ham araç tablosundan toplu tahmin. Önce özellik matrisi üretilir."""
        X, _ = build_feature_matrix(vehicles)
        return self.predict_from_features(X)

    def predict_from_features(self, X: pd.DataFrame) -> np.ndarray:
        """Hazır özellik matrisinden tahmin (değerlendirmede ayrılan test seti için).

        Gecikme negatif olamayacağından tahminler 0'a kırpılır.
        """
        self._require_trained()
        preds = self._booster.predict(X)
        return np.maximum(0.0, np.asarray(preds, dtype=float))

    def predict(self, vehicle: Vehicle) -> float:
        """Tek bir araç için tahmini gecikmeyi (dakika) döndürür."""
        return float(self.predict_batch(vehicle_to_frame(vehicle))[0])

    def feature_importances(self) -> dict[str, float]:
        """Özellik önem skorları (gain) — 'model neye bakıyor' sorusu için."""
        self._require_trained()
        names = self._booster.feature_name()
        gains = self._booster.feature_importance(importance_type="gain")
        return {name: float(gain) for name, gain in zip(names, gains)}

    def save(self, path: str) -> None:
        """Eğitilmiş modeli LightGBM yerel metin biçiminde kaydeder."""
        self._require_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(path)

    @classmethod
    def load(cls, path: str) -> "DelayPredictor":
        """Kaydedilmiş modeli yükler ve bir DelayPredictor'a sarar."""
        instance = cls()
        instance._booster = lgb.Booster(model_file=path)
        instance.feature_names = instance._booster.feature_name()
        return instance

    def _require_trained(self) -> None:
        """Model henüz eğitilmediyse/yüklenmediyse anlamlı bir hata fırlatır."""
        if self._booster is None:
            raise RuntimeError("Model henüz eğitilmedi veya yüklenmedi.")
