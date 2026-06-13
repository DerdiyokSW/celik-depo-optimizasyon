# 07 — Aşama 6: PPO Ajanı (Gymnasium Ortamı + Eğitim)

> **Önkoşul okuma:** `00`, `01`, `03-asama-2-simulasyon-cekirdegi.md`,
> `04-asama-3-gecikme-ml.md`, `05-asama-4-yerlesim-politikalari.md`.
>
> Bu aşama projenin **yıldızı ve tek gerçek öğrenen bileşenidir**. Aşama 3'te
> tasarlanan çekirdek API doğru kurulduğu için bu aşama, çekirdeğin ince bir
> gymnasium kabuğudur — sıfırdan bir simülasyon yazılmaz.

---

## 1. Amaç

Çelik bobinleri depoya, rehandling'i minimize edecek şekilde yerleştirmeyi
**deneme-yanılmayla öğrenen** bir PPO ajanı geliştirmek. Ajan, simülasyon
çekirdeğini bir pekiştirmeli öğrenme ortamı olarak kullanır; öğrendiği politika
`PPOPolicy` olarak diğer üç politikayla aynı arayüzden değerlendirilir.

## 2. Neden PPO (ve neden Q-Learning değil)

Depo durumu uzayı kombinatoryal olarak devasadır; klasik Q-Learning her durum
için tablo tutar ve bu uzayda bellek olarak çöker. PPO bir derin pekiştirmeli
öğrenme algoritmasıdır: politikayı bir sinir ağıyla temsil eder, tablo tutmaz.
Ayrıca PPO'nun "proximal" (clipped) hedef fonksiyonu, politikanın bir güncellemede
çok keskin sapmalar yapmasını engeller — eğitim kararlılığı yüksektir. Detaylı
gerekçe `00-proje-felsefesi.md` §3-4'tedir.

## 3. Üretilecek dosyalar (kod)

```
src/rl/
├── __init__.py
├── warehouse_env.py   # WarehouseEnv(gymnasium.Env) — simülatör sarmalayıcı
├── observation.py      # WarehouseState/bobin -> gözlem tensörü dönüşümü
├── reward.py           # ödül fonksiyonu (ayrı dosya — netlik ve test için)
├── curriculum.py       # zorluk kademelendirme (curriculum learning)
└── train_ppo.py        # MaskablePPO eğitim betiği (main)
```

Ayrıca `src/policies/ppo_policy.py` iskeleti bu aşamada doldurulur.

## 4. Gymnasium ortamı (`warehouse_env.py`)

Ortam, `WarehouseSimulator`'ı sarar. Çekirdek zaten `pending_coil`,
`valid_actions`, `apply_placement` ilkellerini sunduğu için ortam incedir.

```python
class WarehouseEnv(gymnasium.Env):
    """Çelik bobin yerleştirmeyi pekiştirmeli öğrenme ortamı olarak sunar.

    Bir 'adım' = bir bobinin yerleştirilmesi kararı.
    Bir 'bölüm' (episode) = bir simülasyon senaryosu (ör. 24 saatlik vardiya).
    """
    def __init__(self, config: EnvConfig): ...

    def reset(self, seed=None, options=None) -> tuple[obs, info]:
        """Yeni bir senaryo üretir, simülatörü sıfırlar. info içinde ilk
        action_mask döner."""

    def step(self, action: int) -> tuple[obs, reward, terminated, truncated, info]:
        """action (0..143) bir SlotCoord'a çevrilir, apply_placement çağrılır,
        ödül hesaplanır. info içinde güncel action_mask döner."""

    def action_masks(self) -> np.ndarray:
        """MaskablePPO'nun çağırdığı metot: 144 uzunlukta boolean dizi;
        geçerli konumlar True. sim.valid_actions()'tan üretilir."""
```

## 5. Action space ve action masking (kritik)

**Action space:** `Discrete(n_zones * n_bays * n_layers) = Discrete(144)`. Her
indeks bir `SlotCoord`'a birebir eşlenir (`index -> (zone, bay, layer)`).

**Action masking — projenin önemli teknik kararı:** Geçersiz konumlar (dolu
slot, fizik ihlali) ajana **ceza verilerek değil, hiç sunulmayarak** ele alınır.
`sb3-contrib` kütüphanesinin **MaskablePPO** algoritması kullanılır; her adımda
`action_masks()` ile geçerli eylemler bildirilir, ajan yalnızca onlar arasından
seçer.

Neden ceza değil maskeleme: Geçersiz hamleyi cezalandırmak, ajanın öğrenme
kapasitesinin büyük kısmını "yasak hamle yapma"yı öğrenmeye harcamasına yol
açar. Maskeleme bu sorunu kökten çözer; ödül yalnızca **gerçek hedefe**
(rehandling, yükleme süresi) odaklanabilir.

## 6. Observation space (`observation.py`)

Gözlem bir `gymnasium.spaces.Dict`'tir; SB3'te `MultiInputPolicy` ile işlenir.
Tüm sayısal alanlar yaklaşık `[0,1]` aralığına normalize edilir (sinir ağı
eğitimi için şart).

| Anahtar | Şekil | İçerik |
|---|---|---|
| `warehouse` | (4, 12, 3, 3) | Her konum için: doluluk (0/1), normalize ağırlık, normalize aciliyet |
| `pending_coil` | (8,) | Tip (one-hot 3), normalize ağırlık, maks kat, normalize aciliyet, normalize tahmini gecikme, normalize deadline yakınlığı |
| `global` | (3,) | Depo doluluk oranı, kuyruk uzunluğu (normalize), simülasyon zamanı oranı |

```python
def build_observation(sim: WarehouseSimulator, delay_model: DelayPredictor) -> dict:
    """Simülatörün anlık durumundan PPO gözlem sözlüğünü üretir.
    Tahmini gecikme, bekleyen bobinin aracı için delay_model'den alınır —
    böylece PPO da Aşama 3'ün ML çıktısını girdi olarak kullanır (hibrit yapı).
    """
```

> Not: `pending_coil` içine **tahmini gecikme**nin konması, hibrit mimarinin
> can alıcı noktasıdır — PPO kararını ML'in tahminine dayandırır.

## 7. Ödül fonksiyonu (`reward.py`) — en kritik tasarım

Ödül fonksiyonu ayrı bir dosyada, ayrı test edilebilir biçimde tutulur. Üç
bileşenden oluşur. Tüm ağırlıklar, adım başına ödülü kabaca `[-2, 2]`
aralığında tutacak şekilde ayarlanır (PPO için ödül ölçeklemesi önemlidir).

**Bileşen 1 — Anlık yönlendirme ödülü (yoğun, küçük).**
Yerleştirme kalitesinin bir vekil göstergesi. Öğrenmenin erken evresinde ajana
yön verir; küçük tutulur ki ajan gerçek hedefi değil vekili optimize etmesin.
```
guidance = W_FIT * (placement_fit - 0.5) * 2     # ~[-1, 1], merkezlenmiş
         + W_AFF * affinity_term                  # küçük + bonus
```
(`placement_fit` ve `affinity_term`, Aşama 4 `scoring.py`'den yeniden kullanılır.)

**Bileşen 2 — Gerçekleşen sevkiyat sinyali (asıl hedef).**
`apply_placement` dönen `StepResult` içinde, o adımda gerçekleşen sevkiyatların
ürettiği gerçek rehandling ve mesafe vardır. Asıl ödül budur:
```
realized = - W_REH  * rehandling_delta
           - W_DIST * normalized_distance_delta
```

**Bileşen 3 — Bölüm sonu terminal ödülü.**
Bölüm bitince, ajanın toplam performansı bir referansla (ör. rastgele yerleşimin
o senaryodaki rehandling'i) kıyaslanır:
```
terminal = W_TERM * (baseline_rehandling - final_rehandling) / max(baseline_rehandling, 1)
```

**Toplam:** `reward = guidance + realized` (+ bölüm sonunda `terminal`).

**Kredi ataması:** Bir yerleştirmenin yol açtığı rehandling çoğu zaman çok adım
sonra gerçekleşir. Bu gecikmiş ödül sorununu PPO'nun değer fonksiyonu (value
function) ve GAE (Generalized Advantage Estimation) mekanizması çözer — zaten
PPO'yu seçmemizin sebeplerinden biri budur. `guidance` terimi yalnızca erken
öğrenmeyi hızlandıran, potansiyel-tabanlı ödül şekillendirmeye yakın bir yardımcı
sinyaldir; ağırlığı düşük tutulur ki nihai politikayı saptırmasın.

> **Geçersiz hamle cezası YOKTUR** — geçersiz eylemler action masking ile zaten
> elenir. Ödül fonksiyonu fizik ihlalini hiç görmez.

## 8. Bölüm (episode) tanımı

- Bir bölüm = bir simülasyon senaryosu (varsayılan 24 saatlik vardiya).
- `reset()` her seferinde tohumu değişen yeni bir senaryo üretir — ajanın geniş
  bir dağılıma maruz kalması (genelleme/Katman 2 adaptasyon) için şarttır.
- `terminated`: simülasyon ufku doldu veya bekleyen bobin kalmadı.
- `truncated`: maksimum adım sınırına ulaşıldı (güvenlik).

## 9. Curriculum learning (`curriculum.py`)

PPO'nun yakınsamasını kolaylaştırmak için zorluk kademeli artırılır:

1. **Kolay:** Düşük olay hızı (8 olay/saat), az bobin, peak olayı yok.
2. **Orta:** Olay hızı 12/saat, ara sıra peak.
3. **Zor:** Olay hızı 20/saat, sık peak, yoğun kuyruk.

Ajan bir kademede belirli bir performans eşiğini geçince bir üst kademeye
geçer. Bu, doğrudan zor senaryoda eğitime kıyasla daha kararlı bir öğrenme
eğrisi verir.

## 10. Eğitim (`train_ppo.py`)

`stable-baselines3` + `sb3-contrib` kullanılır. Tekerlek yeniden icat edilmez;
yazdığımız kısım **ortam, gözlem, ödül ve curriculum**'dur — algoritma hazır.

- **Algoritma:** `MaskablePPO`, `MultiInputPolicy`.
- **Paralel ortam:** `SubprocVecEnv` ile birden çok ortam (ör. 8) aynı anda —
  veri toplama hızlanır.
- **Başlangıç hiperparametreleri** (CV/deneyle ince ayar yapılır):
  `learning_rate=3e-4`, `n_steps=2048`, `batch_size=512`, `gamma=0.995`,
  `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.005`, `vf_coef=0.5`.
- **Loglama:** TensorBoard (`runs/ppo_warehouse`) — ödül eğrisi, bölüm uzunluğu,
  politika/değer kaybı izlenir.
- **Hedef:** Kaynak izin verdiğince uzun eğitim (mertebe olarak milyonlarca
  adım). Eğitim sonunda model `models/ppo_warehouse.zip`'e kaydedilir.
- **Periyodik değerlendirme:** Eğitim sırasında `EvalCallback` ile ajan, sabit
  bir doğrulama senaryo kümesinde ölçülür; en iyi model ayrıca saklanır.

## 11. PPOPolicy'nin doldurulması (`src/policies/ppo_policy.py`)

Aşama 4'teki iskelet burada gerçeklenir:

```python
class PPOPolicy(PlacementPolicy):
    def decide(self, coil, sim) -> SlotCoord:
        """Eğitilmiş PPO modeliyle karar verir:
        1) build_observation ile gözlemi üret,
        2) sim.valid_actions()'tan action mask üret,
        3) model.predict(obs, action_masks=mask) ile eylem indeksini al,
        4) indeksi SlotCoord'a çevirip döndür.
        """
```

Böylece PPO, değerlendirme hattında (Aşama 7) diğer üç politikayla **birebir
aynı arayüzden** kıyaslanır.

## 12. "Dinamik öğrenme" — periyodik yeniden eğitim kancası

`00-proje-felsefesi.md` §4 Katman 3: Eğitim altyapısı, biriken yeni veriyle
**warm-start** (mevcut modelden devam) yeniden eğitime izin verecek şekilde
yazılır. `train_ppo.py`, opsiyonel bir `--resume-from <model>` parametresi alır.
Bu, Aşama 7'deki "periyodik yeniden eğitim deneyi"nin altyapısıdır.

## 13. Kabul kriterleri

1. `WarehouseEnv` gymnasium API'sine uyar; `gymnasium.utils.env_checker` ve
   maskeleme kontrolünü geçer.
2. `action_masks()` çıktısı `sim.valid_actions()` ile birebir tutarlıdır;
   maskelenmiş hiçbir eylem seçilemez.
3. `python -m src.rl.train_ppo` eğitimi başlatır, TensorBoard log üretir,
   model kaydeder.
4. Ödül eğrisi eğitim boyunca **yukarı yönlü bir eğilim** gösterir (öğrenme
   kanıtı); bölüm başına rehandling düşer.
5. Eğitilmiş `PPOPolicy`, `simulator.run` ile koşar ve geçerli kararlar üretir.
6. `--resume-from` ile warm-start yeniden eğitim çalışır.
7. Determinizm: sabit tohumla eğitim tekrarlanabilir (RL'de tam determinizm
   zor olabilir; en azından tohum sabitlenir ve sapma raporlanır).

## 14. Testler (`tests/test_rl/`)

- `test_env_api`: `WarehouseEnv` env_checker'ı geçer.
- `test_action_mask_consistency`: `action_masks()` ile `valid_actions()` aynı
  konum kümesini gösterir.
- `test_index_slot_mapping`: `index -> SlotCoord -> index` dönüşümü kayıpsız.
- `test_observation_shape`: Gözlem sözlüğü tanımlı `Dict` uzayına uyar; tüm
  değerler normalize aralıkta.
- `test_reward_no_invalid_penalty`: Ödül fonksiyonu fizik ihlali terimi içermez.
- `test_reward_scale`: Tipik adımlarda ödül kabaca `[-2, 2]` aralığında.
- `test_reward_signs`: Rehandling arttığında ödül azalır; iyi `placement_fit`
  ödülü artırır.

## 15. Risk notu

PPO eğitimi zaman ve ince ayar gerektirir. Eğitim beklenen seviyeye ulaşmazsa
proje **çökmez**: `PlacementPolicy` arayüzü sayesinde sistem üç baseline ile
çalışmaya devam eder ve Aşama 7 karşılaştırması PPO'nun o anki gerçek
performansını **dürüstçe** raporlar. Hedef PPO'nun ML-destekli sezgiseli
geçmesidir; geçemezse bu da geçerli bir bilimsel sonuçtur ve savunmada böyle
sunulur (`09-savunma-notlari.md`).
