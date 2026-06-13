# 04 — Aşama 3: Gecikme Tahmin ML Modeli

> **Önkoşul okuma:** `00-proje-felsefesi.md`, `01-veri-sozlesmesi.md`,
> `02-asama-1-veri-uretici.md`.
>
> Bu aşama Aşama 2'den bağımsız geliştirilebilir; tek girdisi
> `vehicles_12m.parquet`'tir. Çıktısı, Aşama 4 ve 6'nın kullanacağı eğitilmiş
> bir tahmin modelidir.

---

## 1. Amaç

Bir aracın **gerçek varış gecikmesini** (dakika) tahmin eden gözetimli bir
regresyon modeli geliştirmek. Bu, projedeki belirsizliğin ölçülebilir bir
sinyale çevrildiği yerdir. Yerleşim politikaları bu tahmini kullanarak bobinleri
doğru kata yerleştirir (geç gelmesi beklenen aracın bobini daha derine konabilir).

## 2. Girdi / Çıktı

**Girdi:** `data/vehicles_12m.parquet` — `01-veri-sozlesmesi.md` §5 şeması.
**Çıktı:** `models/delay_model.txt` (eğitilmiş LightGBM modeli) + bir metrik
raporu (`runs/delay_model_report.json`).

## 3. Üretilecek dosyalar (kod)

```
src/ml/
├── __init__.py
├── features.py       # özellik mühendisliği — Vehicle/df -> özellik matrisi
├── delay_model.py    # DelayPredictor sınıfı (eğitim + tahmin + kaydet/yükle)
├── train.py          # model karşılaştırma + final model eğitimi (main)
└── evaluate.py       # metrik raporu üretimi
```

## 4. Özellik mühendisliği (`features.py`)

Modelin yalnızca **varıştan önce bilinebilecek** özellikleri kullanması şarttır.

**Veri sızıntısı uyarısı (savunmada sorulur):** `actual_arrival` ve
`delay_minutes` asla özellik olarak kullanılamaz — bunlar hedefin kendisidir.
Model yalnızca aşağıdaki, araç yola çıkmadan bilinen özellikleri görür.

| Özellik | Kaynak | İşlem |
|---|---|---|
| `distance_km` | doğrudan | sayısal, olduğu gibi |
| `carrier_quality_score` | doğrudan | sayısal, 0..1 |
| `traffic_index` | doğrudan | sayısal, 0..1 |
| `weather` | doğrudan | one-hot kodlama (CLEAR/RAIN/SNOW) |
| `vehicle_type` | doğrudan | one-hot kodlama (TRUCK/TRAIN/SHIP) |
| `planned_hour` | `planned_arrival`'tan | saat bileşeni (0..23) |
| `weekday` | `planned_arrival`'tan | haftanın günü (0..6) |
| `month` | `planned_arrival`'tan | ay (1..12) |

**Hedef değişken:** `delay_minutes`.

```python
def build_feature_matrix(vehicles: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Ham araç tablosundan model girdisi (X) ve hedefi (y) üretir.

    Zaman damgasından saat/gün/ay çıkarır, kategorik alanları one-hot kodlar.
    Hedef sızıntısına yol açabilecek (actual_arrival, delay_minutes) sütunları
    X'e ASLA dahil etmez. Dönüş: (X, y).
    """
```

## 5. DelayPredictor (`delay_model.py`)

```python
class DelayPredictor:
    """Araç gecikmesini tahmin eden gözetimli model sarmalayıcısı.

    Eğitim, tahmin ve kalıcılaştırmayı tek bir temiz arayüz altında toplar.
    Aşama 4 ve 6 yalnızca predict() / predict_batch() metotlarını kullanır.
    """
    def train(self, vehicles: pd.DataFrame) -> None:
        """Modeli eğitir. İç işleyiş: özellik matrisi üret, train/test böl,
        LightGBM eğit. Hiperparametreler train.py'de CV ile seçilmiştir."""

    def predict(self, vehicle: SteelVehicle) -> float:
        """Tek bir araç için tahmini gecikmeyi (dakika) döndürür."""

    def predict_batch(self, vehicles: pd.DataFrame) -> np.ndarray:
        """Toplu tahmin (değerlendirme ve eğitim verimliliği için)."""

    def feature_importances(self) -> dict[str, float]:
        """Özellik önem skorları — savunmada 'model neye bakıyor' sorusu için."""

    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "DelayPredictor": ...
```

## 6. Model seçimi ve eğitim (`train.py`)

Tek bir modele körü körüne gidilmez; karşılaştırmalı bir seçim yapılır
(savunmada "neden bu model" sorusuna cevap olur).

1. **Karşılaştırılan modeller:** Lineer Regresyon (taban çizgisi), Karar Ağacı,
   Random Forest, LightGBM.
2. **Değerlendirme:** 5-katlı çapraz doğrulama (5-fold CV) ile her modelin
   ortalama MAE'si ölçülür.
3. **Hiperparametre ayarı:** Kazanan model (beklenen: LightGBM) için CV
   üzerinden temel hiperparametreler ayarlanır (`n_estimators`, `learning_rate`,
   `max_depth`, `num_leaves`).
4. **Final eğitim:** Seçilen model tüm eğitim verisiyle eğitilir, test setinde
   son kez ölçülür, `models/delay_model.txt`'e kaydedilir.

## 7. Değerlendirme metrikleri (`evaluate.py`)

Test setinde hesaplanır ve `runs/delay_model_report.json`'a yazılır:

- **MAE** (Mean Absolute Error) — ortalama mutlak hata, dakika.
- **RMSE** (Root Mean Squared Error) — büyük hataları cezalandırır, dakika.
- **R²** — açıklanan varyans oranı.
- **≤15 dk hata oranı** — tahminlerin yüzde kaçı 15 dakika içinde isabetli.

**Beklenen performans hakkında dürüst not:** Veri üretici (`02 §5`) gecikmeye
standart sapması ~8 dakika olan indirgenemez bir Gaussian gürültü ekler. Bu,
RMSE için yaklaşık 8 dakikalık bir teorik alt sınır demektir — *hiçbir model*
bunun altına inemez. Dolayısıyla R²'nin 1.0 olmaması bir kusur değil,
gerçekçiliğin sonucudur ve savunmada böyle açıklanmalıdır. Eldeki durum
raporundaki sayılar (MAE 8.4, R² 0.81) **göstermeliktir**; gerçek sayılar bu
hattın çıktısıdır ve onları taklit etmek yasaktır.

## 8. Kabul kriterleri

1. `python -m src.ml.train` çalışır, dört modeli karşılaştırır, kazananı eğitir
   ve `models/delay_model.txt`'i üretir.
2. `runs/delay_model_report.json` dört metriği de içerir.
3. `DelayPredictor.load()` kaydedilen modeli yükler ve `predict()` tek bir araç
   için makul (negatif olmayan, mertebe olarak doğru) bir gecikme döndürür.
4. **Sızıntı kontrolü:** `build_feature_matrix` çıktısı `actual_arrival` veya
   `delay_minutes` içermez (test ile doğrulanır).
5. `feature_importances()` anlamlı sonuç verir — `weather`, `carrier_quality`
   ve `distance` yüksek önemde çıkmalı (veri üretici örüntüsüyle tutarlı).
6. Determinizm: sabit `random_state` ile eğitim tekrarlanabilir.

## 9. Testler (`tests/test_ml/`)

- `test_no_leakage`: Özellik matrisi sızıntı sütunlarını içermez.
- `test_feature_shape`: One-hot sonrası beklenen sütunlar mevcut.
- `test_predict_range`: Tahminler negatif değil; mertebe makul.
- `test_save_load`: Kaydedilip yüklenen model aynı tahmini üretir.
- `test_importance_sanity`: En önemli 3 özellik arasında `weather`/`carrier`/
  `distance` türevleri yer alır.

## 10. Bu aşama bittiğinde elde olan

Eğitilmiş, ölçülmüş, kaydedilmiş bir gecikme tahmin modeli. Aşama 4'teki
`MLHeuristicPolicy` ve Aşama 6'daki PPO ortamı bu modeli girdi olarak kullanmaya
hazırdır.
