# 02 — Aşama 1: Veri Üretici (Data Generator)

> **Önkoşul okuma:** `00-proje-felsefesi.md`, `01-veri-sozlesmesi.md`.
> Bu aşama sistemin ilk yapı taşıdır; sonraki tüm aşamalar bu çıktıyı kullanır.

---

## 1. Amaç

Gerçek endüstriyel veriye erişim olmadığı için, sistemin ihtiyaç duyduğu tüm
veriyi **sentetik ama gerçekçi** biçimde üretmek. Kritik nokta: veri rastgele
değildir — içine makine öğrenmesinin yakalayabileceği **gizli örüntüler**
bilinçle gömülür. Özellikle araç gecikmeleri, hava/firma/mesafe gibi
faktörlere bağlı *öğrenilebilir* bir yapı taşır.

## 2. Girdi / Çıktı

**Girdi:** Yalnızca konfigürasyon parametreleri (üretilecek bobin sayısı,
sipariş sayısı, ay sayısı, `seed`).

**Çıktı:** `data/` klasörüne 5 dosya — `01-veri-sozlesmesi.md` §8'deki şemalara
birebir uygun: `coils.parquet`, `vehicles_12m.parquet`, `orders.parquet`,
`warehouse_config.json`, `initial_state.json`.

## 3. Üretilecek dosyalar (kod)

```
src/data/
├── __init__.py
├── config.py            # GeneratorConfig dataclass — tüm parametreler
├── coil_generator.py     # bobin üretimi
├── vehicle_generator.py  # araç + gecikme örüntüsü üretimi
├── order_generator.py    # sipariş üretimi, bobin-araç eşleştirme
├── layout_generator.py   # depo konfigürasyonu + bozuk başlangıç durumu
└── generate_all.py       # hepsini sırayla çalıştıran giriş noktası (main)
```

## 4. Sınıf ve fonksiyon imzaları

İmzalar bağlayıcıdır. Her fonksiyon, mantığını Türkçe açıklayan bir docstring
ve karmaşık satırlarda Türkçe yorum içerir.

```python
# config.py
@dataclass
class GeneratorConfig:
    """Veri üretiminin tüm ayarlanabilir parametrelerini tutar.

    Tek bir yerden yönetim sağlar; deney tekrarlanabilirliği `seed` ile garanti
    edilir (aynı seed -> aynı veri seti).
    """
    n_coils: int = 5000
    n_orders: int = 1200
    n_months: int = 12          # araç geçmişi kaç aylık üretilecek
    n_vehicles: int = 3600
    seed: int = 42


# coil_generator.py
def generate_coils(config: GeneratorConfig) -> pd.DataFrame:
    """Sentetik çelik bobin envanterini üretir.

    Her bobinin tipi (sıcak/soğuk/galvaniz) gerçekçi bir dağılımla seçilir;
    ağırlık, çap, genişlik ve maks istif katı tipe göre `01-veri-sozlesmesi.md`
    §2'deki aralıklardan örneklenir. Üretim zamanı `n_months` boyunca yayılır.

    Dönüş: SteelCoil şemasına uygun DataFrame (location ve order_id henüz boş).
    """


# vehicle_generator.py
def generate_vehicles(config: GeneratorConfig) -> pd.DataFrame:
    """12 aylık geçmiş araç kaydını, ÖĞRENİLEBİLİR gecikme örüntüsüyle üretir.

    Gecikme rastgele değildir; `compute_delay_minutes()` ile faktörlere bağlı
    hesaplanır. Bu sayede Aşama 3'teki tahmin modeli anlamlı sinyal yakalar.
    """


def compute_delay_minutes(
    weather: Weather,
    carrier_quality_score: float,
    distance_km: float,
    traffic_index: float,
    vehicle_type: VehicleType,
    rng: np.random.Generator,
) -> float:
    """Bir aracın gecikmesini (dakika) faktörlere bağlı olarak hesaplar.

    Gizli örüntü buradadır — Aşama 3 modeli tam olarak bu ilişkiyi öğrenmeye
    çalışacaktır. Formül §5'te tanımlanmıştır. Sonuç 0'ın altına düşmez.
    """


# order_generator.py
def generate_orders(
    config: GeneratorConfig,
    coils: pd.DataFrame,
    vehicles: pd.DataFrame,
) -> pd.DataFrame:
    """Siparişleri üretir ve her siparişe bobin + araç atar.

    Bobin-sipariş bağını ÇİFT YÖNLÜ kurar: coils tablosundaki `order_id` da
    güncellenir. Referans bütünlüğü (§01 doğrulama kuralı 2) sağlanmalıdır.
    """


# layout_generator.py
def generate_warehouse_config(config: GeneratorConfig) -> dict:
    """WarehouseLayout şemasına uygun depo konfigürasyonunu üretir (4x12x3)."""


def generate_initial_state(
    config: GeneratorConfig,
    coils: pd.DataFrame,
    layout: dict,
) -> dict:
    """Bilinçli olarak BOZUK bir başlangıç yerleşimi üretir.

    Bobinlerin bir kısmı, sevkiyat aciliyetiyle çelişen konumlara yerleştirilir
    (erken sevkiyat -> alt kat, geç sevkiyat -> üst kat gibi). Bu, optimizasyon
    politikalarının iyileştirme kabiliyetini ölçecek referans zemindir.
    İstif süreklilik ve ağırlık kuralları (§01 kural 4-5) yine de İHLAL EDİLMEZ;
    yerleşim 'kötü' olabilir ama 'fiziksel olarak geçersiz' olamaz.
    """


# generate_all.py
def main(config: GeneratorConfig | None = None) -> None:
    """Tüm veri setini sırayla üretir, doğrular ve data/ klasörüne yazar."""
```

## 5. Gecikme örüntüsü — iş kuralı (aşamanın kalbi)

`compute_delay_minutes` aşağıdaki mantığı uygular. Amaç **doğrusal olmayan ama
öğrenilebilir** bir ilişki kurmaktır:

```
base_delay        = 5 dakika (sabit taban)

weather_effect    = { CLEAR: 0, RAIN: 25, SNOW: 60 }   (dakika)

carrier_effect    = 90 * (1 - carrier_quality_score)
                    # düşük sicilli firma -> büyük gecikme; 0..90 dk

distance_effect   = 0.04 * distance_km
                    # mesafe arttıkça gecikme artar; 2..48 dk

traffic_effect    = 40 * traffic_index                  (0..40 dk)

# DOĞRUSAL OLMAYAN ETKİLEŞİM: kötü hava + uzun mesafe birlikte daha da kötü
interaction       = 0.0008 * distance_km * weather_effect

type_effect       = { TRUCK: 1.0, TRAIN: 0.6, SHIP: 1.4 } çarpanı
                    # gemi lojistiği daha oynak, tren daha stabil

noise             = Gaussian(mean=0, std=8)             (dakika)

delay = type_effect * (base_delay + weather_effect + carrier_effect
        + distance_effect + traffic_effect + interaction) + noise

delay = max(0, delay)   # negatif gecikme yok
```

Bu formül kod içinde sabit olarak gömülmez; `vehicle_generator.py` içinde
adlandırılmış sabitler (`WEATHER_DELAY_MAP` vb.) olarak tutulur ki hem okunaklı
hem ayarlanabilir olsun. Her terimin üstüne ne yaptığını anlatan Türkçe yorum konur.

> Önemli: Tahmin modeli (Aşama 3) bu formülü *bilmez*; sadece girdi-çıktı
> çiftlerinden öğrenir. `noise` terimi mükemmel tahmini imkânsız kılar — bu
> gerçekçidir ve modelin R²'sinin neden 1.0 olmadığını açıklar.

## 6. Diğer iş kuralları

- **Bobin tipi dağılımı:** Soğuk hadde ağırlıklı bir tesis varsayılır —
  yaklaşık `COLD_ROLLED %45, GALVANIZED %35, HOT_ROLLED %20`.
- **Sipariş–araç eşleşmesi:** Her sipariş tek bir araçla karşılanır; aracın
  `target_logistics_line`'ı siparişteki bobinlerin gideceği zone ile uyumlu seçilir
  (affinity ödülünün anlamlı olabilmesi için).
- **Sipariş büyüklüğü:** Bir sipariş 1–8 bobin içerir; aracın kapasitesi
  aşılmaz (`max_weight_capacity_ton`).
- **Öncelik dağılımı:** `NORMAL %70, HIGH %20, URGENT %10`.
- **Zaman tutarlılığı:** `deadline > production_time`; `planned_arrival`
  sipariş `deadline`'ından önce olacak şekilde seçilir.
- **Tüm rastgelelik** `np.random.default_rng(config.seed)` üzerinden tek bir
  generator ile yapılır — global `random`/`np.random` durumu kullanılmaz.

## 7. Kabul kriterleri

Aşama, aşağıdakilerin tamamı sağlanmadan "bitti" sayılmaz:

1. `python -m src.data.generate_all` çalışır ve 5 dosyayı `data/` altına yazar.
2. Üretilen veri `01-veri-sozlesmesi.md` §10'daki **8 doğrulama kuralının
   tamamını** geçer (doğrulama, yazımdan önce kod içinde otomatik koşar).
3. Aynı `seed` ile iki kez çalıştırıldığında **birebir aynı** dosyalar üretilir
   (determinizm).
4. `vehicles_12m.parquet` içinde gecikme ile faktörler arasındaki ilişki
   gözlemlenebilir: ör. SNOW kayıtlarının ortalama gecikmesi CLEAR'dan belirgin
   yüksektir; düşük `carrier_quality_score` yüksek gecikmeyle ilişkilidir.
   (Bunu doğrulayan basit bir grup-ortalaması kontrolü yazılır.)
5. `initial_state.json` fiziksel olarak geçerlidir (istif/ağırlık ihlali yok)
   ama bilinçli olarak optimumdan uzaktır.

## 8. Testler (`tests/test_data/`)

- `test_coil_ranges`: Her bobin tipinin ağırlık/çap/genişlik değerleri
  `01-veri-sozlesmesi.md` §2 aralıklarında.
- `test_id_uniqueness`: Tüm kimlikler benzersiz.
- `test_referential_integrity`: Bobin–sipariş–araç bağları tutarlı (çift yönlü).
- `test_delay_non_negative`: Hiçbir `delay_minutes` negatif değil.
- `test_delay_pattern`: SNOW ortalama gecikmesi > RAIN > CLEAR; düşük sicilli
  firma ortalama gecikmesi yüksek sicilliden büyük.
- `test_determinism`: Aynı seed iki çalıştırmada aynı DataFrame hash'i üretir.
- `test_initial_state_valid`: Başlangıç yerleşimi istif süreklilik ve ağırlık
  kurallarını ihlal etmez.

## 9. Bu aşama bittiğinde elde olan

Çalıştırılabilir, doğrulanmış, tekrarlanabilir bir veri seti. Aşama 2
(simülasyon çekirdeği) ve Aşama 3 (gecikme ML) artık başlayabilir.
