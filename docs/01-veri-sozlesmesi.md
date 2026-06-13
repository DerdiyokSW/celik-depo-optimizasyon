# 01 — Veri Sözleşmesi (Data Contract)

> Modüller arası **tüm** veri alışverişi bu dosyadaki şemalara birebir uyar.
> Bir alan veya tip değişecekse önce bu dosya güncellenir, sonra kod. Bu sözleşme
> sistemin "ortak dili"dir; veri üretici, simülasyon, ML ve RL bunu paylaşır.
>
> Tüm veri modelleri `src/domain/` altında Python `@dataclass` veya `Enum` olarak
> tanımlanır. Tabular veri setleri `pandas.DataFrame` ile taşınır ve `parquet`
> olarak saklanır.

---

## 1. Enum tanımları

```
CoilType      = { HOT_ROLLED, COLD_ROLLED, GALVANIZED }   # sıcak/soğuk/galvaniz
QualityClass  = { A, B }            # A = yüzeyi hassas, B = standart
CoilStatus    = { IN_PRODUCTION, PENDING_PLACEMENT, STORED, LOADED, DISPATCHED }
VehicleType   = { TRUCK, TRAIN, SHIP }
Weather       = { CLEAR, RAIN, SNOW }
OrderPriority = { NORMAL, HIGH, URGENT }
OrderStatus   = { OPEN, IN_PROGRESS, FULFILLED, CANCELLED }
LogisticsLine = { SHIP_1, SHIP_2, TRAIN_A, TRUCK_DOCK }   # sevkiyat hatları
EventType     = { NEW_ORDER, CANCEL_ORDER, VEHICLE_DELAY, PRIORITY_CHANGE, PEAK_LOAD }
```

## 2. SteelCoil — çelik bobin

Sistemin yerleştirdiği temel nesne. `src/domain/coil.py`.

| Alan | Tip | Açıklama / Aralık |
|---|---|---|
| `coil_id` | str | Benzersiz kimlik, ör. `COIL-080001` |
| `coil_type` | CoilType | Bobin tipi |
| `weight_ton` | float | Ağırlık (ton). Tipe göre aralık aşağıda |
| `width_mm` | int | Genişlik (mm) |
| `diameter_mm` | int | Dış çap (mm) |
| `quality_class` | QualityClass | Yüzey hassasiyeti |
| `max_stack_layer` | int | Üstüne çıkılabilecek maks kat (tipten türetilir) |
| `production_time` | datetime | Üretim bandından çıkış anı |
| `order_id` | str \| None | Ait olduğu sipariş; henüz atanmadıysa None |
| `status` | CoilStatus | Yaşam döngüsü durumu |
| `location` | SlotCoord \| None | Depodaki konumu; depoda değilse None |
| `urgency_score` | float | 0..1, sevkiyat aciliyeti (simülasyonda hesaplanır) |
| `rehandled` | bool | Sevkiyatta yer değiştirdi mi (engelleyici olarak taşındı). Çalışma-zamanı bayrağı; veri setine yazılmaz, görselleştirmede işaretlenir. Varsayılan False |

> Not: `max_stack_layer` bobinin fiziksel istif limitidir (tip türevli); depo
> yalnızca 2 katlı olduğundan etkin istif her tipte en fazla 2'dir.

**Tipe göre fiziksel parametreler** (veri üretici bu aralıkları kullanır):

| CoilType | weight_ton | width_mm | diameter_mm | max_stack_layer | tipik quality |
|---|---|---|---|---|---|
| HOT_ROLLED  | 20.0 – 30.0 | 1000 – 2000 | 1500 – 2100 | 2 | çoğunlukla B |
| COLD_ROLLED | 12.0 – 22.0 | 800 – 1500  | 1200 – 1600 | 3 | A/B karışık |
| GALVANIZED  | 10.0 – 18.0 | 700 – 1300  | 1000 – 1500 | 3 | çoğunlukla A |

> Not: Tüm ağırlıklar 10–30 ton bandındadır; bu, resmî proje özetiyle tutarlıdır.
> Slab ve boru/profil bilinçli olarak kapsam dışıdır — rehandling problemi
> özünde bobin piramit istifleme problemidir.

## 3. SlotCoord — depo konumu

Depodaki tek bir istif konumunu temsil eder. `src/domain/warehouse.py`.

| Alan | Tip | Açıklama / Aralık |
|---|---|---|
| `zone` | int | Bölge indeksi, 0..7 |
| `bay` | int | Bölge içi sıra, 0..35 |
| `layer` | int | Dikey kat, 0..1 (0 = zemin) |

Toplam konum sayısı: `8 × 36 × 2 = 576`.

> Boyut gerekçesi: gerçek bir çelik deposu büyük, dikdörtgen bir holdür. 8 bölge ×
> 66 sıra geniş bir taban verir; dikey istif yalnızca 2 kattır (Borçelik'te üst üste
> istif yoktur, ancak 2 katlı tutmak rehandling/yer değişimini görsel olarak
> gösterebilmek için bilinçli bir modelleme tercihidir).

## 4. WarehouseLayout — depo geometrisi

Depo yapısının değişmez tanımı. `warehouse_config.json` dosyasından yüklenir.

| Alan | Tip | Açıklama |
|---|---|---|
| `n_zones` | int | 8 |
| `n_bays` | int | Zone başına 36 |
| `n_layers` | int | Stack başına 2 |
| `zone_logistics` | dict[int, LogisticsLine] | Her zone'un hizmet ettiği sevkiyat hattı (4 hat, hat başına 2 zone) |
| `zone_max_weight_ton` | dict[int, float] | Zone başına toplam tonaj limiti |
| `entry_point` | (int, int) | Üretim bandı çıkışı, mesafe hesabı için (bay=0, zone=0 referansı) |

**Modelleme varsayımı — istif (önemli, savunmada açıklanmalı):**
Gerçek piramit istif fiziği bay ekseni boyunca komşuluk gerektirir (alt sıradaki
iki bobinin yuvasına üst bobin oturur). Bu projede istif, **stack-içi sütun
soyutlamasıyla** modellenir: her `(zone, bay)` hücresi en fazla 2 katlık bir
dikey sütundur. Piramit fiziği şu iki kuralla temsil edilir:

1. **Süreklilik:** `layer L`'ye bobin koymak için `layer L-1` dolu olmalıdır
   (havada bobin olamaz).
2. **Ağırlık azalması:** `layer L`'deki bobin, `layer L-1`'deki bobinden hafif
   olmalıdır (ağır alta).

Bu, lisans düzeyi coil-stacking modellerinde standart ve savunulabilir bir
soyutlamadır. Gerçek çapraz-bay piramit fiziği ileride bir iyileştirme olarak
eklenebilir; bu projenin kapsamında değildir.

## 5. Vehicle — sevkiyat aracı

Bobinleri depodan alan araç. Gecikme tahmin modelinin ana veri kaynağı.
`src/domain/vehicle.py`.

| Alan | Tip | Açıklama / Aralık |
|---|---|---|
| `vehicle_id` | str | Benzersiz kimlik |
| `vehicle_type` | VehicleType | TRUCK / TRAIN / SHIP |
| `max_weight_capacity_ton` | float | TRUCK ~25, TRAIN ~60, SHIP ~120 |
| `planned_arrival` | datetime | Planlanan varış (ETA) |
| `actual_arrival` | datetime | Gerçekleşen varış (canlı simülasyonda varışta açığa çıkar) |
| `delay_minutes` | float | `actual - planned`, dakika. **ML hedef değişkeni**. ≥ 0 |
| `carrier_id` | str | Lojistik firma kimliği |
| `carrier_quality_score` | float | 0..1, firma sicili (yüksek = güvenilir) |
| `weather` | Weather | Varış günü hava durumu |
| `distance_km` | float | Kat edilecek mesafe, 50 – 1200 |
| `traffic_index` | float | 0..1, yol yoğunluğu |
| `target_logistics_line` | LogisticsLine | Hangi hatta/zone'a hizmet ettiği |

## 6. Order — sevkiyat siparişi

Bir araca yüklenecek bobin grubu. `src/domain/order.py`.

| Alan | Tip | Açıklama |
|---|---|---|
| `order_id` | str | Benzersiz kimlik, ör. `ORD-001234` |
| `vehicle_id` | str | Siparişi karşılayan araç |
| `coil_ids` | list[str] | Yüklenecek bobinlerin kimlikleri |
| `deadline` | datetime | Son sevk zamanı |
| `priority` | OrderPriority | NORMAL / HIGH / URGENT |
| `status` | OrderStatus | OPEN / IN_PROGRESS / FULFILLED / CANCELLED |

**Tutarlılık kuralı:** `order.vehicle_id` geçerli bir araca; `coil_ids` içindeki
her kimlik geçerli bir bobine işaret etmelidir. Bir bobinin `order_id`'si onu
içeren siparişin `order_id`'si ile eşleşmelidir (çift yönlü bağ).

## 7. Event — dinamik olay

Simülasyon sırasında zaman ekseninde üretilen olay. `src/domain/event.py`.

| Alan | Tip | Açıklama |
|---|---|---|
| `timestamp` | float | Simülasyon saatinde olayın anı (saat cinsinden) |
| `event_type` | EventType | Olay tipi |
| `payload` | dict | Olaya özgü veri (aşağıda) |

Olay tiplerine göre `payload` içeriği:

- `NEW_ORDER`: `{ "order": Order }` — kuyruğa yeni sipariş.
- `CANCEL_ORDER`: `{ "order_id": str }` — mevcut sipariş iptali.
- `VEHICLE_DELAY`: `{ "vehicle_id": str, "extra_delay_minutes": float }`.
- `PRIORITY_CHANGE`: `{ "order_id": str, "new_priority": OrderPriority }`.
- `PEAK_LOAD`: `{ "burst_factor": float }` — kriz/zirve senaryosu tetikleyici.

## 8. Veri seti dosyaları

Veri üretici (Aşama 1) aşağıdaki dosyaları `data/` altına yazar.

| Dosya | İçerik | Varsayılan boyut |
|---|---|---|
| `coils.parquet` | Tüm bobin envanteri | 5.000 bobin |
| `vehicles_12m.parquet` | 12 aylık geçmiş araç kaydı (ML eğitimi için) | 3.600 kayıt |
| `orders.parquet` | Sipariş tablosu | 1.200 sipariş |
| `warehouse_config.json` | Depo geometrisi (WarehouseLayout) | — |
| `initial_state.json` | Bilinçli "bozuk" başlangıç yerleşimi | — |

**`initial_state.json` neden "bozuk"?** Başlangıç durumu bilinçle optimize
edilmemiştir: bazı erken sevkiyat bobinleri erişimi zor alt katlara, bazı geç
sevkiyat bobinleri üst katlara konmuştur. Bu, geliştirilen politikaların
*iyileştirme kabiliyetini* ölçmek için referans (baseline) zemindir.

## 9. Modüller arası veri akışı

```
Veri Üretici  ──► coils.parquet, vehicles_12m.parquet, orders.parquet,
                  warehouse_config.json, initial_state.json
                       │
   ┌───────────────────┼────────────────────────┐
   ▼                    ▼                        ▼
Gecikme ML        Simülasyon Çekirdeği      Event Generator
(vehicles_12m'i   (config + initial_state   (orders/vehicles
 okur, model      ile depoyu kurar)          dağılımından olay üretir)
 üretir)               │                        │
   │                   ▼                        │
   │            Yerleşim Politikaları ◄──────────┘
   │            (PlacementPolicy)
   └──► gecikme tahmini ──► MLHeuristicPolicy / PPOPolicy girdisi
```

## 10. Doğrulama kuralları

Veri üretici çıktıyı yazmadan önce, simülasyon çekirdeği veriyi yüklerken
aşağıdaki kontroller otomatik yapılır. İhlal varsa istisna fırlatılır.

1. **Kimlik benzersizliği:** Tüm `coil_id`, `vehicle_id`, `order_id` benzersiz.
2. **Referans bütünlüğü:** Her `order.coil_ids` ve `order.vehicle_id` geçerli
   nesnelere işaret eder; bobin–sipariş bağı çift yönlü tutarlıdır.
3. **Konum geçerliliği:** Her `SlotCoord` için `0 ≤ zone < 8`, `0 ≤ bay < 36`,
   `0 ≤ layer < 2`.
4. **İstif süreklilik:** Dolu bir `layer L` için `layer L-1` de dolu olmalı.
5. **Ağırlık kuralı:** Bir stack içinde yukarı çıkıldıkça ağırlık azalmalı.
6. **Kapasite:** Bir zone'daki bobinlerin toplam ağırlığı `zone_max_weight_ton`'u
   aşamaz.
7. **Gecikme işareti:** `delay_minutes ≥ 0` (negatif gecikme yok; erken varış 0).
8. **Tip–kat tutarlılığı:** `max_stack_layer` bobin tipinin izin verdiği değerle
   eşleşmeli (HOT_ROLLED=2, diğerleri=3).
