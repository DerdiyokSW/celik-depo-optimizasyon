# 05 — Aşama 4: Yerleşim Politikaları

> **Önkoşul okuma:** `00`, `01`, `03-asama-2-simulasyon-cekirdegi.md`,
> `04-asama-3-gecikme-ml.md`.
>
> **Kilometre taşı:** Bu aşama bittiğinde çalışan bir sistem + 3 baseline
> politika eldedir. Proje bu noktada savunulabilir durumdadır; PPO (Aşama 6)
> bunun üzerine eklenen yıldızdır.

---

## 1. Amaç

Simülasyon çekirdeğinin koşturacağı yerleştirme politikalarını üretmek. Tüm
politikalar tek bir ortak arayüzü (`PlacementPolicy`) uygular — bu, projenin
stabilite garantisidir: çekirdek hangi politikanın takılı olduğunu bilmez,
politikalar birbirinin yerine takılıp çıkarılabilir.

Bu aşamada üç politika tamamlanır: Rastgele, Klasik Sezgisel, ML-destekli
Sezgisel. Dördüncüsü (PPOPolicy) burada yalnızca iskelet olarak tanımlanır,
Aşama 6'da doldurulur.

## 2. Üretilecek dosyalar (kod)

```
src/policies/
├── __init__.py
├── base.py               # PlacementPolicy soyut taban sınıfı
├── scoring.py            # ortak skorlama fonksiyonları (sezgiseller paylaşır)
├── random_policy.py      # RandomPolicy
├── heuristic_policy.py   # HeuristicPolicy (klasik sezgisel)
├── ml_heuristic_policy.py# MLHeuristicPolicy (gecikme tahmini ile beslenen)
└── ppo_policy.py         # PPOPolicy — Aşama 6'da doldurulacak iskelet
```

## 3. PlacementPolicy arayüzü (`base.py`)

```python
class PlacementPolicy(ABC):
    """Tüm yerleştirme politikalarının uyduğu ortak sözleşme.

    Simülasyon çekirdeği yalnızca bu arayüzü tanır; somut politikayı bilmez.
    Dört politika (Random/Heuristic/MLHeuristic/PPO) bunu uygular ve
    birbirinin yerine takılabilir — değerlendirme bu sayede tek hatla yapılır.
    """

    @abstractmethod
    def decide(self, coil: SteelCoil, sim: WarehouseSimulator) -> SlotCoord:
        """Bekleyen bobin için bir yerleştirme konumu seçer.

        Dönen konum, sim.valid_actions() listesinden biri OLMAK ZORUNDADIR;
        politika geçerli konumlar dışına çıkamaz.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Raporlama ve grafiklerde kullanılan kısa ad (ör. 'Heuristic')."""
```

Politika `sim` üzerinden şunlara erişir: `sim.valid_actions()` (geçerli
konumlar), `sim.state` (depo durumu), ve bobin→sipariş→araç bağını çözmek için
çekirdeğin sunduğu arama metotları (`sim.order_of(coil)`, `sim.vehicle_of(order)`).

## 4. RandomPolicy (`random_policy.py`)

En basit baseline. Geçerli konumlar arasından `seed`'li bir generator ile
**düzgün dağılımla** birini seçer. Akıllı hiçbir şey yapmaz; diğer politikaların
ne kadar değer kattığını ölçmek için alt sınırdır.

```python
class RandomPolicy(PlacementPolicy):
    def __init__(self, seed: int = 0): ...
    def decide(self, coil, sim) -> SlotCoord:
        """sim.valid_actions() içinden rastgele bir konum döndürür."""
```

## 5. Ortak skorlama (`scoring.py`)

Klasik ve ML-destekli sezgisel aynı skorlama mantığını paylaşır; aralarındaki
tek fark **aciliyetin nasıl hesaplandığıdır** (§7). Skorlama fonksiyonları saf
ve test edilebilirdir.

```python
def coil_urgency(hours_to_dispatch: float) -> float:
    """Bobinin sevkiyatına kalan süreden 0..1 aciliyet skoru üretir.
    Sevkiyat ne kadar yakınsa skor o kadar yüksektir.
    urgency = clip(1 - hours_to_dispatch / URGENCY_HORIZON, 0, 1)
    """

def slot_accessibility(slot: SlotCoord, layout: WarehouseLayout) -> float:
    """Bir konumdan bobin almanın ne kadar kolay olduğunu 0..1 ile ölçer.
    Üst kat = erişilebilir (üstünde yük yok); zone çıkışına yakın = erişilebilir.
    accessibility = W_LAYER * layer_score + W_DIST * exit_proximity_score
    """

def placement_fit(urgency: float, accessibility: float) -> float:
    """Aciliyet ile erişilebilirliğin ne kadar ÖRTÜŞTÜĞÜ (0..1).
    Acil bobin erişilebilir konuma, acil olmayan derine konmalı.
    fit = 1 - |urgency - accessibility|
    """

def affinity_bonus(coil: SteelCoil, slot: SlotCoord, sim) -> float:
    """Bobin, kendi lojistik hattına hizmet eden zone'a konursa +1, değilse 0.
    J4n1k affinity fikrinin uyarlaması: aynı araca gidecek bobinler aynı zone'da.
    """

def score_slot(coil, slot, urgency, sim, layout) -> float:
    """Bir aday konumun toplam skoru.
    score = W_FIT * placement_fit + W_AFFINITY * affinity_bonus
            - W_DIST * normalized_entry_distance
    Ağırlık sabitleri scoring.py'de adlandırılmış ve yorumlanmış tutulur.
    """
```

## 6. HeuristicPolicy (`heuristic_policy.py`) — klasik sezgisel

Açıklanabilir, hızlı, kuralı net bir politika. Akışı:

1. Bobinin sevkiyat saatini **planlanan varış zamanından** (ML kullanmadan)
   al; kalan süreden `coil_urgency` hesapla.
2. `sim.valid_actions()` ile geçerli konumları al.
3. Her geçerli konum için `score_slot` hesapla.
4. En yüksek skorlu konumu döndür (eşitlikte `seed`'li deterministik kırıcı).

```python
class HeuristicPolicy(PlacementPolicy):
    def decide(self, coil, sim) -> SlotCoord:
        """Aciliyet–erişilebilirlik örtüşmesi + affinity ile en iyi konumu seçer.
        Aciliyet, planlanan varış zamanına dayanır (gecikme tahmini KULLANILMAZ).
        """
```

Bu politika literatürdeki ABC-sınıfı / öncelik tabanlı yerleştirme
stratejilerinin uyarlamasıdır; affinity bileşeni J4n1k'ten esinlenir ama
fitness "zaman-öncelikli katmanlı yerleştirme" olarak yeniden formüle edilmiştir.

## 7. MLHeuristicPolicy (`ml_heuristic_policy.py`) — ML-destekli sezgisel

`HeuristicPolicy` ile **birebir aynı skorlama**yı kullanır; tek fark aciliyetin
hesaplanma biçimidir. Bu, ML'in net katkısını izole etmek için bilinçli bir
tasarımdır — iki politika arasındaki tek değişken gecikme tahminidir.

1. Bobinin aracını bul, `DelayPredictor.predict(vehicle)` ile gecikmeyi tahmin et.
2. **Etkin sevkiyat zamanı** = `planlanan_varış + tahmini_gecikme`.
3. Aciliyeti bu etkin zamana göre hesapla — geç gelmesi beklenen aracın bobini
   daha az acil görünür, dolayısıyla daha derine yerleştirilebilir.
4. Geri kalan akış `HeuristicPolicy` ile aynı.

```python
class MLHeuristicPolicy(PlacementPolicy):
    def __init__(self, delay_model: DelayPredictor): ...
    def decide(self, coil, sim) -> SlotCoord:
        """Aciliyeti, ML ile düzeltilmiş etkin sevkiyat zamanından hesaplar;
        gerisi HeuristicPolicy ile aynıdır. ML katkısı bu tek farkta görülür.
        """
```

## 8. PPOPolicy iskeleti (`ppo_policy.py`)

Bu aşamada yalnızca arayüze uyan bir iskelet bırakılır; gövdesi Aşama 6'da
eğitilmiş PPO modeliyle doldurulur. Böylece değerlendirme hattı (Aşama 7) dört
politikayı da baştan tanır.

```python
class PPOPolicy(PlacementPolicy):
    def __init__(self, model_path: str | None = None):
        """Eğitilmiş PPO modelini yükler. model_path None ise Aşama 6'ya kadar
        decide() NotImplementedError fırlatır."""
    def decide(self, coil, sim) -> SlotCoord: ...
```

## 9. İş kuralları

- Bir politika **asla** geçersiz konum döndürmez; her zaman `valid_actions()`
  içinden seçer. `valid_actions()` boşsa (depo dolu) çekirdek bunu taşma olayı
  olarak yönetir — politika değil.
- Skorlama ağırlık sabitleri (`W_FIT`, `W_AFFINITY`, `W_DIST`, `W_LAYER`,
  `URGENCY_HORIZON`) `scoring.py` tepesinde adlandırılmış sabitler olarak durur;
  sihirli sayı kod içine gömülmez.
- Tüm politikalar deterministiktir: aynı `seed` + aynı durum → aynı karar.

## 10. Kabul kriterleri

1. Dört politika da `PlacementPolicy` arayüzünü uygular; `simulator.run(policy,
   horizon)` her biriyle çalışır (PPOPolicy hariç — o Aşama 6'da).
2. Aynı senaryoda (aynı seed) üç politika koşturulduğunda **beklenen sıralama**
   gözlenir: rehandling açısından `Rastgele > Klasik Sezgisel ≥ ML-destekli
   Sezgisel`. (Kesin sayılar değil, sıralama beklenir.)
3. Hiçbir politika geçersiz konum döndürmez (test ile doğrulanır).
4. ML-destekli sezgisel ile klasik sezgisel arasındaki **tek kod farkı**
   aciliyet hesabıdır — skorlama paylaşılır.
5. Determinizm: her politika sabit seed ile tekrarlanabilir sonuç verir.

## 11. Testler (`tests/test_policies/`)

- `test_policy_returns_valid_slot`: Her politikanın kararı `valid_actions()`
  içindedir.
- `test_random_uniform`: RandomPolicy yeterince çeşitli konum üretir.
- `test_heuristic_prefers_match`: Acil bobin için erişilebilir konum, acil
  olmayan için derin konum tercih edilir (kontrollü senaryo).
- `test_affinity`: Eşit koşulda bobin, lojistik hattına uygun zone'u tercih eder.
- `test_ml_vs_heuristic_difference`: Gecikme tahmini yüksek olan bir araç için
  MLHeuristicPolicy, HeuristicPolicy'den farklı (daha derin) bir konum seçer.
- `test_ordering`: Küçük bir senaryoda rehandling sıralaması beklenen yönde.

## 12. Bu aşama bittiğinde elde olan

Uçtan uca çalışan, üç baseline politikalı bir sistem. Gecikme tahmininin
yerleşime kattığı değer ölçülebilir hale gelmiştir. Sıradaki aşama (5) bunu
görsel bir dashboard'a bağlar; Aşama 6 ise öğrenen PPO ajanını ekler.
