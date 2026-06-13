# 03 — Aşama 2: Simülasyon Çekirdeği ve Event Generator

> **Önkoşul okuma:** `00-proje-felsefesi.md`, `01-veri-sozlesmesi.md`,
> `02-asama-1-veri-uretici.md`.
>
> Bu aşama projenin **en kritik mimari kararıdır**. Buradaki API tasarımı,
> Aşama 6'daki PPO ortamının (gymnasium) ince bir sarmalayıcı mı yoksa baştan
> yazım mı olacağını belirler. Çekirdek doğru tasarlanırsa hem değerlendirme
> koşucusu hem PPO ortamı aynı ilkelleri kullanır.

---

## 1. Amaç

Arayüzsüz (headless), deterministik bir depo simülasyon çekirdeği kurmak. Bu
çekirdek:

- Depo 3B durumunu (4×12×3 = 144 konum) tutar.
- Fizik/istif kısıtlarını uygular ve **geçerli yerleştirme konumlarını** sorgular
  (bu sorgu Aşama 6'da MaskablePPO'nun action mask'i olur).
- Sevkiyat anında **rehandling** sayımını ve diğer metrikleri hesaplar.
- Zamanı olaylarla (event-driven) ilerletir.
- Hem politika-güdümlü (değerlendirme) hem dışarıdan-güdümlü (RL) modda
  çalışabilir.

## 2. Üretilecek dosyalar (kod)

```
src/simulation/
├── __init__.py
├── warehouse_state.py   # WarehouseState — 3B depo durumu, yerleştirme/sorgu
├── constraints.py        # fizik/istif kısıt kontrolleri (saf fonksiyonlar)
├── metrics.py            # metrik hesaplama (rehandling, mesafe, süre, doluluk)
├── dispatch.py           # sevkiyat/retrieval mantığı + rehandling sayımı
├── event_generator.py    # EventGenerator — Poisson süreçli olay akışı
└── simulator.py          # WarehouseSimulator — ana orkestratör, zaman döngüsü
```

## 3. Çekirdek API tasarımı (en önemli bölüm)

Simülatör, hem değerlendirme koşucusunun hem RL ortamının kullanacağı **ortak
ilkelleri** sunar. Bu sayede Aşama 6 yalnızca bir gymnasium kabuğu olur.

```python
class WarehouseSimulator:
    """Depo simülasyonunun ana orkestratörü.

    İki kullanım modu:
      1) Politika-güdümlü: run() ile bir PlacementPolicy uçtan uca koşturulur.
      2) Dışarıdan-güdümlü: pending_coil() / valid_actions() / apply_placement()
         ilkelleri RL ortamı tarafından adım adım çağrılır.
    Her iki mod da AYNI ilkelleri kullanır — kod tekrarı yoktur.
    """

    def reset(self) -> None:
        """Depoyu initial_state.json'daki başlangıç durumuna geri alır."""

    # --- Dışarıdan-güdümlü mod ilkelleri (RL ortamı bunları kullanır) -------
    def pending_coil(self) -> SteelCoil | None:
        """Yerleştirilmeyi bekleyen sıradaki bobin (yoksa None)."""

    def valid_actions(self) -> list[SlotCoord]:
        """Bekleyen bobinin TÜM kısıtları sağlayarak konabileceği konumlar.
        RL'de bu liste action mask'e dönüştürülür — geçersiz eylem hiç sunulmaz."""

    def apply_placement(self, slot: SlotCoord) -> StepResult:
        """Bekleyen bobini verilen konuma yerleştirir, zamanı bir sonraki karar
        noktasına kadar ilerletir, bu arada gerçekleşen olayları işler.
        Dönüş: StepResult (metrik değişimi, gerçekleşen olaylar, done bayrağı)."""

    def advance_to_next_decision(self) -> None:
        """Yeni bir yerleştirme kararı gerekene kadar olayları işleyip zamanı
        ilerletir (sevkiyatlar, iptaller vb. burada gerçekleşir)."""

    def is_done(self) -> bool:
        """Simülasyon ufku doldu mu / işlenecek bobin kaldı mı?"""

    @property
    def metrics(self) -> SimulationMetrics:
        """Biriken metrikler (rehandling, mesafe, süre, doluluk...)."""

    @property
    def state(self) -> WarehouseState:
        """Anlık depo durumu (gözlem üretimi ve görselleştirme için)."""

    # --- Politika-güdümlü mod (değerlendirme koşucusu bunu kullanır) --------
    def run(self, policy: "PlacementPolicy", horizon_hours: float) -> SimulationMetrics:
        """Verilen politikayı simülasyon ufku boyunca uçtan uca koşturur.
        İç döngü: pending_coil -> policy.decide -> apply_placement."""
```

```python
@dataclass
class StepResult:
    """Tek bir yerleştirme adımının sonucu."""
    rehandling_delta: int        # bu adımda eklenen rehandling sayısı
    distance_delta: float        # bu adımda eklenen vinç mesafesi (m)
    events_occurred: list[Event] # adım sırasında gerçekleşen olaylar
    done: bool
```

## 4. WarehouseState — depo durumu

`warehouse_state.py`. 144 konumu verimli tutar; RL bölümü her bölümde sıfırlanıp
binlerce kez kopyalanacağı için **kopyalanması ucuz** olmalıdır.

```python
class WarehouseState:
    def is_empty(self, slot: SlotCoord) -> bool: ...
    def coil_at(self, slot: SlotCoord) -> SteelCoil | None: ...
    def place(self, coil: SteelCoil, slot: SlotCoord) -> None: ...
    def remove(self, coil: SteelCoil) -> SteelCoil: ...
    def valid_slots(self, coil: SteelCoil) -> list[SlotCoord]:
        """Bu bobinin konabileceği geçerli konumlar (constraints.can_place ile)."""
    def fill_ratio(self) -> float:
        """Dolu konum / toplam konum oranı (0..1)."""
    def stack_height(self, zone: int, bay: int) -> int:
        """Bir (zone,bay) sütununda kaç kat dolu."""
    def snapshot(self) -> np.ndarray:
        """Durumu sayısal tensöre çevirir (Aşama 6 gözlem uzayının temeli)."""
    def copy(self) -> "WarehouseState":
        """Derin ama ucuz kopya — RL bölüm sıfırlamaları için."""
```

## 5. constraints.py — fizik kuralları (saf fonksiyonlar)

Yan etkisiz, test edilmesi kolay saf fonksiyonlar. Bir bobin bir konuma
konabilir mi?

```python
def can_place(state: WarehouseState, coil: SteelCoil, slot: SlotCoord) -> bool:
    """Tüm kısıtları kontrol eder; hepsi sağlanırsa True."""

def placement_violations(state, coil, slot) -> list[str]:
    """Hangi kuralların ihlal edildiğini liste olarak döndürür (loglama/hata ayıklama)."""
```

Kontrol edilen kurallar (`01-veri-sozlesmesi.md` ile birebir):

1. **Boşluk:** Hedef konum boş olmalı.
2. **İstif süreklilik:** `layer L > 0` ise `layer L-1` dolu olmalı.
3. **Ağırlık azalması:** `layer L > 0` ise alttaki bobin bu bobinden ağır olmalı.
4. **Maks kat:** `layer < coil.max_stack_layer`.
5. **Zone kapasitesi:** Konum bobini alınca zone toplam tonajı limiti aşmamalı.

## 6. dispatch.py — rehandling sayımı (metriklerin kalbi)

Bir aracın varışında siparişinin bobinleri depodan alınır. Rehandling burada
sayılır. Algoritma:

```
dispatch_order(state, order):
    hedef_bobinler = order.coil_ids içinden STORED durumda olanlar
    hedefleri (zone, bay) sütununa göre grupla
    rehandling = 0
    mesafe = 0
    her hedef-içeren sütun için:
        sütunu YUKARIDAN AŞAĞIYA tara (layer 2 -> 0):
            o kattaki bobin hedef mi?
              EVET  -> üretken hamle: bobini sütundan al, araç rıhtımına taşı
                       mesafe += crane_distance(slot -> zone_cikis)
              HAYIR -> altında hâlâ alınacak bir hedef var mı?
                  VAR   -> ENGELLEYİCİ bobin: başka geçerli boş konuma taşı
                           rehandling += 1
                           mesafe += crane_distance(slot -> yeni_slot)
                  YOK   -> dokunma (hamle yok)
    return DispatchResult(rehandling, mesafe, hamleler)
```

Önemli incelik: Bir bobinin üstündeki başka bir bobin de **aynı sevkiyatın
hedefi** ise, onu kaldırmak rehandling değil üretken hamledir. Rehandling
yalnızca *erişimi açmak için zorunlu olarak* oynatılan, kendisi sevk edilmeyen
bobinler için sayılır.

`crane_distance` Manhattan mesafesidir (`metrics.py`); `01-veri-sozlesmesi.md`
§6 J4n1k entegrasyonu buradan gelir.

## 7. metrics.py — metrik hesaplama

```python
def crane_distance(a: SlotCoord, b: SlotCoord) -> float:
    """İki konum arası Manhattan mesafesi (bay ve zone ekseni); metre cinsinden
    ölçeklenir. Vinç hareket maliyetinin temsilidir."""

@dataclass
class SimulationMetrics:
    """Bir simülasyon koşusunun tüm performans metrikleri."""
    rehandling_count: int = 0
    total_crane_distance_m: float = 0.0
    total_loading_time_min: float = 0.0
    final_fill_ratio: float = 0.0
    n_placements: int = 0
    n_dispatches: int = 0
    decision_times_ms: list[float] = field(default_factory=list)
```

**Yükleme süresi modeli:** Her vinç hamlesi sabit bir hazırlık süresi + mesafeyle
orantılı bir süre alır:
`loading_time_min += CRANE_SETUP_MIN + distance_m * CRANE_TIME_PER_M`.
Sabitler `metrics.py` içinde adlandırılmış ve yorumlanmış olarak tutulur.

## 8. event_generator.py — dinamik olaylar

```python
class EventGenerator:
    """Simülasyon zaman ekseninde dinamik olaylar üretir.

    Olay zamanlaması Poisson süreciyle (üstel dağılımlı aralıklar) belirlenir;
    olay tipi ise verilen olasılık karışımından seçilir.
    """
    def __init__(self, rate_per_hour: float = 12.0,
                 type_mix: dict[EventType, float] | None = None,
                 seed: int = 7): ...

    def stream(self, horizon_hours: float) -> Iterator[Event]:
        """Ufuk boyunca zaman-sıralı olay akışı üretir (generator)."""

    def trigger_peak(self, burst_factor: float = 5.0) -> Event:
        """Kriz/zirve senaryosu: olay hızını ve sipariş aciliyetini geçici
        olarak artıran bir PEAK_LOAD olayı üretir (Knapp entegrasyonu)."""
```

Varsayılan olay karışımı (rapordaki dağılımla uyumlu):
`NEW_ORDER 0.55, VEHICLE_DELAY 0.25, CANCEL_ORDER 0.10, PRIORITY_CHANGE 0.10`.
Varsayılan hız 12 olay/saat. Değerlendirmede hız parametre olarak değiştirilip
(8/12/20 olay/saat) dayanıklılık analizi yapılır (Aşama 7).

## 9. Zaman modeli

Simülasyon **sürekli zaman**lıdır (float, saat cinsinden). Bir olay kuyruğu
tutulur; kuyrukta üç tür "zamanlanmış şey" vardır: (i) bobin üretim/giriş anları,
(ii) araç varış anları, (iii) EventGenerator'ın ürettiği olaylar. Simülatör
kuyruğu zaman sırasıyla işler. Bir bobin girişi yerleştirme kararı gerektirir;
bir araç varışı sevkiyatı (dispatch) tetikler.

## 10. İş kuralları

- **VEHICLE_DELAY olayı**, hedef aracın `actual_arrival`'ını öteler — yani
  sevkiyat daha geç olur, bu arada üstüne bobin yığılabilir. Politikanın
  gecikme tahminini kullanmasının değeri tam burada ortaya çıkar.
- **CANCEL_ORDER**, siparişi `CANCELLED` yapar; bobinleri tekrar atanabilir hale
  gelir (yeniden planlama / Katman 1 adaptasyon).
- **Determinizm:** Simülatör ve EventGenerator ayrı `seed` alır; aynı seed +
  aynı politika = birebir aynı metrikler.
- **Geçersiz yerleştirme:** `apply_placement` geçersiz bir slot alırsa istisna
  fırlatır — çünkü geçerli mod, çağıranın önce `valid_actions()`'a bakmasıdır.

## 11. Kabul kriterleri

1. `WarehouseSimulator` hem `run(policy, horizon)` hem
   `pending_coil/valid_actions/apply_placement` ilkelleriyle çalışır.
2. `valid_actions()` yalnızca `constraints.can_place` true dönen konumları
   içerir — hiçbir geçersiz konum sızmaz.
3. Aynı seed + aynı politika iki koşuda birebir aynı `SimulationMetrics` üretir.
4. Rehandling sayımı elle kurulmuş küçük senaryolarda doğrulanır (§12 testleri).
5. 24 saatlik bir senaryo (~320 olay) makul sürede (saniyeler) tamamlanır;
   `decision_times_ms` ölçülür ve loglanır.
6. `reset()` sonrası depo birebir `initial_state.json`'a döner.

## 12. Testler (`tests/test_simulation/`)

- `test_constraints`: Her kısıt kuralı için pozitif/negatif örnek
  (boş olmayan slot, kat atlama, ağır-üste-hafif-alta, maks kat aşımı, kapasite).
- `test_valid_actions_subset`: `valid_actions()` çıktısının her elemanı
  `can_place` testini geçer.
- `test_rehandling_known_case`: Elle kurulan istif senaryosunda rehandling
  sayısı beklenen değere eşit (ör. hedefin üstünde 2 engelleyici → 2 rehandling).
- `test_rehandling_productive_not_counted`: Hedefin üstündeki bobin de hedefse
  rehandling sayılmaz.
- `test_determinism`: Aynı seed iki koşuda aynı metrikler.
- `test_reset`: `reset()` sonrası durum başlangıçla aynı.
- `test_event_stream_ordered`: EventGenerator olayları zaman-sıralı üretir;
  hız parametresi olay sayısını beklenen mertebede etkiler.

## 13. Bu aşama bittiğinde elde olan

Çalışan, test edilmiş, deterministik bir simülasyon çekirdeği. Henüz akıllı bir
politika yoktur ama çekirdek, herhangi bir `PlacementPolicy`'yi koşturmaya ve
RL ortamı tarafından sarılmaya hazırdır.
