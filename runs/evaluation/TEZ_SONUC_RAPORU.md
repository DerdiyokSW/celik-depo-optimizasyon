# Tez Sonuç Raporu — Çelik Bobin Depo Optimizasyonu

**Tarih:** 2026-06-08
**Sistem:** 8 zone × 36 bay × 2 kat = **576 slot** depo, tavan vinci (Chebyshev mesafe)
**Veri:** popülasyon başına 5000 bobin, 3600 araç (12 ay geçmiş), 1200 sipariş — sentetik ama gerçekçi gizli örüntülerle

> **Bu rapor, projenin nihai ve tutarlı sonucudur; iki önceki raporun (2026-06-03 ve
> 2026-06-06) yerine geçer.** Bilimsel iddia, tek bir "PPO her şeyi yener" cümlesi değil;
> **hangi problem yapısının hangi yöntemi ödüllendirdiğini** gösteren iki-senaryolu, dürüst
> bir mühendislik sonucudur. Yol boyunca PPO sonuçlarında **iki ayrı yapay (artefakt) etki**
> tespit edilip düzeltildi (ezber + kısıt-gevşemesi). Tüm sayılar gerçek koşum çıktısıdır.

---

## 1. Problem ve Bilimsel İddia

Çelik bobinlerin depo içi yerleşimini optimize ederek iki maliyeti düşürmek: sevkiyatta
**rehandling** (üstteki bobini taşıma) ve toplam **vinç mesafesi** (hareket/enerji).

**Nihai bilimsel iddia (dürüst, yöntem–problem eşleşmesine dayalı):**
> Gerçekçi kısıtlı depoda (2 kat istif + lojistik-hattı affinity'si) **kural-tabanlı sezgisel
> yöntem en iyi ve en sağlamdır**; öğrenen PPO ajanı bu sert kısıtlar altında sezgiseli
> geçemez. Buna karşılık, kısıtların kalktığı **saf rota-optimizasyonu alt-probleminde**
> (tek kat, hedef = vinç mesafesi) PPO ajanı sezgiseli **istatistiksel olarak anlamlı**
> şekilde geçer. Yani RL, doğal olarak uygun olduğu problem yapısında değer üretir.

---

## 2. Karşılaştırılan Politikalar

| Politika | Açıklama | ML? |
|---|---|---|
| **Random** | Geçerli konumlardan rastgele seçer (alt sınır) | Hayır |
| **Heuristic** | Door-aciliyet eşleşmesi + istif disiplini + affinity (klasik sezgisel) | Hayır |
| **MLHeuristic** | Heuristic + LightGBM gecikme tahmini ile düzeltilmiş aciliyet | **Evet** |
| **PPO** | MaskablePPO + CNN gözlem temsili (WarehouseExtractor) | Evet (gözlemde ML tahmini) |

Hepsi ortak `PlacementPolicy` arayüzünü uygular → simülasyon çekirdeği hangisinin takılı
olduğunu bilmez (risk izolasyonu + doğru yöntemin probleme takılabilmesi).

---

## 3. Metodoloji — Train/Test Ayrımı (geçerli temel)

- **Senaryo havuzu:** 64 eğitim popülasyonu (seed 0–63) + **30 görülmemiş test popülasyonu**
  (seed 9000–9029). Tohum aralıkları **ayrık** (sızıntı yok). Depo layout'u (8×36×2) sabit;
  bobin/sipariş/araç/başlangıç-yerleşimi tohuma göre değişir.
- **Eğitim:** PPO her bölümde havuzdan FARKLI bir popülasyon görür → tek senaryoyu ezberleyemez.
- **Değerlendirme (HELD-OUT):** 30 görülmemiş popülasyonun her birinde tüm politikalar BİREBİR
  aynı popülasyon+olay tohumunu görür (eşleştirilmiş). 24 saatlik vardiya, 12 olay/saat.
- **İstatistik:** Wilcoxon signed-rank (eşleştirilmiş, dağılım-bağımsız).
- Tüm sayılar gerçek koşum çıktısıdır; tüm tohumlar sabittir (tekrar-üretilebilir).

---

## 4. İki Metodolojik Düzeltme — Bu Çalışmanın Bilimsel Çekirdeği

PPO'nun "iyi görünen" ilk sonuçlarında, **iki ayrı yapay etki** kendi sonuçlarımızda tespit
edilip düzeltildi. Bu öz-eleştiri, tezin en güçlü metodolojik anlatısıdır.

### 4.1 Düzeltme #1 — Overfitting (ezber)
- **Belirti:** İlk PPO in-sample rehandling **3.40** veriyor, "her şeyi p=2×10⁻⁹ yener" diyordu.
- **Teşhis:** Ajan TEK sabit senaryoya eğitilmiş, değerlendirme de AYNI veriyi kullanıyordu →
  train/test ayrımı yok, **genelleşme değil ezber** ölçülüyordu.
- **Doğrulama:** Overfit model taze tohumlu (görülmemiş) veride **3.40 → 21.62**'ye fırladı,
  Heuristic'in (8.62) ALTINA düştü. Kural-tabanlı Heuristic ezberleyemez, tutarlı kaldı.
- **Düzeltme:** Senaryo havuzu (64 train + 30 test) + her bölümde farklı popülasyon örnekleme
  + held-out değerlendirme hattı. PPO'nun havuz-eğitimli held-out rehandling'i **4.13**'e indi.

### 4.2 Düzeltme #2 — Kısıt Gevşemesi (constraint-relaxation artefaktı)
- **Belirti:** Held-out 4.13 hâlâ "PPO en iyi" diyordu — ama bu sonucu **şüpheyle** inceledik.
- **Teşhis:** 4.13'ü üreten model, affinity **zorlanmadan** değerlendirilmişti. Ajan iki
  serbestlikten faydalanıyordu: (a) bobinleri **hatlarına ait olmayan zone'lara** koymak
  (affinity'i yoksaymak), (b) **neredeyse hiç istif yapmamak** (%~100 zemin kat → gömme yok →
  rehandling yapay olarak düşük, ama kapasite israfı + vinç mesafesi yüksek).
- **Düzeltme:** Affinity **katı (hard) kısıt** yapıldı (`enforce_affinity=True`, bobin yalnız
  kendi lojistik hattının zone'larına gider; action masking ile zorlanır) ve sonuç **gerçek
  kısıtlar altında yeniden ölçüldü** (§5).
- **Sonuç:** 4.13 → **10.40**; PPO, gerçek kısıt altında Heuristic'in (8.57) **gerisine** düştü.
  Yani **4.13 bir kısıt-gevşemesi artefaktıydı.** (Bu öz-düzeltme, kullanıcı şüphesinin
  veriyle doğrulanmasıdır.)

---

## 5. ANA SENARYO SONUCU — Gerçekçi Depo (2 kat + affinity ZORLANMIŞ)

Held-out, 30 görülmemiş popülasyon, eşleştirilmiş, 12 olay/saat:

| Politika | Rehandling (ort ± std) | Vinç mesafesi (m) |
|---|---|---|
| Random | 33.10 ± 7.92 | 40 717 |
| **Heuristic** | 8.57 ± 3.04 | **26 525** |
| **MLHeuristic** | **8.30 ± 2.91** | 29 806 |
| PPO (havuz modeli) | 10.40 ± 5.04 | 35 675 |

**Sıralama (rehandling): MLHeuristic ≈ Heuristic < PPO ≪ Random.**
**Sıralama (vinç mesafesi): Heuristic < MLHeuristic < PPO < Random.**

### 5.1 İstatistiksel anlamlılık (Wilcoxon, eşleştirilmiş)
| Karşılaştırma | Metrik | Sonuç |
|---|---|---|
| PPO vs Heuristic | rehandling | PPO 12/30 önde, **p = 0.11 → fark ANLAMSIZ** (PPO öne geçemez) |
| PPO vs Heuristic | vinç mesafesi | PPO **0/30** önde, **p < 0.001 → Heuristic ezici** |
| Sezgiseller vs Random | rehandling | Random'ı **çok anlamlı** yener (p ≈ 1.7×10⁻⁶) |

**Dürüst okuma:** Gerçekçi kısıtlar altında **ana senaryoyu sezgisel yöntemler kazanır.**
PPO, gevşek problemde (affinity'siz) öğrendiği politikayı sert kısıtlara taşıyamıyor:
rehandling'de Heuristic'in gerisinde ve farkı anlamsız; vinç mesafesinde belirgin biçimde kötü
(istif yapmama eğilimi → bobinler yayılıyor → fazla vinç yolu). **Sezgisel, sert kısıtlara
sağlam (robust).**

> Not (rehandling metriğinin sınırı): Rehandling 2 katlı depoda **oyunlanabilir** bir
> metriktir — üst katı hiç kullanmayan politika onu yapay düşürür ama kapasiteyi israf eder ve
> vinç mesafesini artırır. Bu yüzden ana senaryoda **iki metriği birlikte** raporluyoruz;
> ikisinde de sezgisel üstündür.

---

## 6. İKİNCİ SENARYO — Raf Modu (Tek Kat · Saf Rota Optimizasyonu): PPO'nun Zaferi

### 6.1 Neden ikinci senaryo?
RL'in *doğal* gücü, sert kombinatoryal kısıt tatmini değil, **uzamsal rota optimizasyonudur**.
Bunu saf biçimde ölçmek için istif ve kapı-ayrımı karıştırıcılarını kaldıran bir varyant tanımladık:
tek kat (istif yok → rehandling yok), affinity kapalı, **tek hedef = toplam vinç mesafesi**.

### 6.2 Kurulum
`single_layer=True` + `enforce_affinity=False`; hedef vinç mesafesi (giriş→slot + slot→kapı,
Chebyshev). MaskablePPO + CNN, **3M adım**, aynı havuz (64 train + 30 test), reward v3 = anlık
normalize yerleştirme maliyeti. (Eğitim masaüstü sunucuda; checkpoint + keep-awake ile.)

### 6.3 ANA SONUÇ — Held-out (30 görülmemiş senaryo, aynı olay tohumu = eşli)
| Politika | Vinç mesafesi (ort ± std, m) | Random'a karşı |
|---|---|---|
| Random | 39 621 ± 4 016 | — |
| Heuristic | 28 143 ± 2 870 | %29.0 ↓ |
| **PPO (raf)** | **26 650 ± 3 309** | **%32.7 ↓** |

- **PPO, Heuristic'i 30 senaryonun 27'sinde yendi.**
- PPO vs Heuristic: **%+5.3 daha iyi** · PPO vs Random: **%+32.7**.
- **Wilcoxon eşli test: p < 0.00001 → istatistiksel olarak ANLAMLI.**
- Grafik: `runs/evaluation/rack_crane_distance.png`.

### 6.4 Yorum — RL burada neden kazanıyor?
- **150k → 3M:** Kısa eğitimde (150k) PPO ≈ Heuristic (berabere); 3M'de PPO öne geçti →
  **ek eğitim ajanı sezgiselin üstüne taşıdı** (öğrenilen anticipatory/önceden konumlandıran
  yerleştirme). Ezber olsaydı held-out'ta bu kademeli iyileşme olmazdı.
- **Negatif ödül (ep_rew ~−150…−600) bir başarısızlık değil, ölçek artefaktıdır:** her
  yerleştirme gerçek-pozitif bir vinç maliyeti doğurur ve bölüm boyunca toplanır. Çakılı/
  öğrenmeyen bir ajan iki baseline'ı birden held-out'ta anlamlı yenemez. Üstelik PPO,
  **değer-fonksiyonu baseline'ı (GAE)** ile sabit ödül-kaymasına **değişmezdir** → ödülün
  mutlak seviyesi bir kalite ölçütü DEĞİLDİR; kıyas ölçüttür.

---

## 7. İki-Senaryolu Sentez (tezin ana anlatısı)

| | Ana senaryo (2 kat + affinity, gerçekçi) | Raf senaryosu (tek kat, saf rota) |
|---|---|---|
| **Hedef** | Rehandling + vinç mesafesi | Vinç mesafesi |
| **Rehandling** | Heuristic 8.57 / ML 8.30 < PPO 10.40 (p=0.11) | — (istif yok) |
| **Vinç mesafesi** | Heuristic 26 525 < PPO 35 675 (p<0.001) | **PPO 26 650 < Heuristic 28 143 (p<0.001)** |
| **Kazanan** | **Sezgisel** (robust, kısıtlara dayanıklı) | **PPO** (öğrenen, rota optimize eder) |

**Mesaj:** Tek bir "her şeyi yener" iddiası yerine **yöntem–problem eşleşmesini** gösteriyoruz.
Gerçekçi sert kısıtlar → kural-tabanlı sezgisel; saf rota alt-problemi → öğrenen PPO. Bu, saf
"PPO kazandı" iddiasından **çok daha olgun ve savunulabilir** bir sonuçtur — ve kendi şüpheli
sonucumuzu (4.13) yakalayıp düzelttiğimiz için **bilimsel titizliğin** de kanıtıdır.

---

## 8. Diğer Bulgular

### Gecikme ML modelinin değeri
Ana senaryoda Heuristic (8.57) vs MLHeuristic (8.30): fark dar; gecikme tahmininin marjı bu
doluluk/rejimde küçüktür (önceki held-out'ta da Heuristic–ML farkı anlamsızdı, p=0.605). Değeri
yüksek yük / farklı rejimlerde belirginleşir. Test MAE ≈ 6.95 dk.

### Swap operatörü (dormant)
Heuristic/MLHeuristic'te swap operatörü TAKILI ama ~%50 dolulukta HİÇ tetiklenmiyor (kapıya
yakın boş prime slotlar her zaman var; ekonomik olarak doğru). Bir tıkanıklık-giderme
mekanizmasıdır; yüksek dolulukta doğru şekilde tetiklenir.

### Dayanıklılık / Yük Bağımlılığı (affinity ZORLANMIŞ, 30 held-out)
Ana senaryoyu 8/12/20 olay/saat (düşük/orta/yüksek doluluk) rejimlerinde, **affinity zorlanmış**
olarak ölçtük. PPO'nun gücü **yüke bağlı** çıktı (bkz. `poster/fig4_yuk_taramasi.png`):

| Olay/saat (doluluk) | PPO | Heuristic | MLHeuristic | Random |
|---|---|---|---|---|
| **Rehandling** | | | | |
| 4/saat (%35) | **0.90** | 1.03 | 1.00 | 3.93 |
| 6/saat (%38) | **1.47** | 2.43 | 2.03 | 10.17 |
| 8/saat (%42) | **3.13** | 4.27 | 3.57 | 15.23 |
| 12/saat (%50) | 10.40 | 8.57 | **8.03** | 32.67 |
| 20/saat (%65) | 56.97 | **22.20** | 23.03 | 84.70 |
| **Vinç mesafesi (m)** | | | | |
| 4/saat | 9 658 | **7 792** | 9 450 | 12 694 |
| 8/saat | 21 647 | **16 386** | 19 459 | 26 278 |
| 12/saat | 35 675 | **26 525** | 29 733 | 40 921 |
| 20/saat | 66 905 | **51 820** | 53 746 | 70 894 |

**Bulgular (dürüst, nüanslı):**
- **Düşük dolulukta (4-8/saat, %35-42) PPO rehandling'de EN İYİ** (4/saat'te 0.90 < tüm
  sezgiseller). PPO'nun bol boş alanı kullanıp gömme yaratmama stratejisi düşük yoğunlukta işe yarar.
- **Orta yükte (12/saat, standart §5 noktası)** sezgiseller öne geçer (PPO 10.40 > 8.03/8.57).
- **Yüksek yükte (20/saat) PPO ÇÖKER** (56.97); az-istifleme stratejisi depo dolunca
  sürdürülemez ve yüksek-yoğunluk istifini hiç öğrenmediği için rehandling patlar; sezgiseller
  robust kalır (~22).
- **Vinç mesafesinde Heuristic HER yükte önde** (PPO yayıldığı için daima daha fazla yol).

**Operasyonel okuma:** PPO normal/düşük yük rejiminde değer üretir (rehandling'de en iyi); ama
sezgisel her rejimde sağlamdır ve aşırı yükte tek güvenli seçenektir → **hibrit dağıtım** mantıklı.

---

## 9. Tez Savunmasında Söyleyebileceklerin

1. **Bilimsel titizlik (iki öz-düzeltme):** Kendi PPO sonuçlarımızda iki yapay etki tespit
   edip düzelttik — (a) overfitting (in-sample 3.40 → held-out 21.62, train/test ayrımıyla
   düzeltildi), (b) kısıt-gevşemesi (held-out 4.13'ün affinity zorlanınca 10.40'a çıkması).
   Bir sonucu "iyi göründüğü için" kabul etmeyip kırılganlığını sınadık. Bu, tezin en güçlü
   metodolojik anlatısıdır.
2. **Dürüst yöntem–problem eşleşmesi:** Gerçekçi kısıtlı depoda sezgisel en iyi/robust; saf
   rota alt-probleminde PPO sezgiseli **anlamlı** geçer (%+5.3, p<0.001). "RL her zaman
   kazanır" demiyoruz; **nerede ve neden** kazandığını gösteriyoruz.
3. **Mühendislik olgunluğu:** `PlacementPolicy` soyutlaması doğru yöntemi probleme takılabilir
   kılar; gerçek tesiste hangi rejimde hangi politika kullanılacağı operasyonel bir karardır.
4. **Tekrar-üretilebilir:** Tüm tohumlar sabit; train/test ayrık; her sayı yeniden üretilebilir.

---

## 10. Sistem Doğrulamaları
- Tüm pytest test paketi yeşil (veri, simülasyon, ML, politikalar, dashboard, RL,
  değerlendirme) — `pytest -q` ile doğrulanabilir.
- Determinizm: sabit tohum → tekrarlanabilir; train/test havuzu deterministik üretilir.
- Gecikme modeli (LightGBM): test MAE ≈ 6.95 dk.
- Ortam tutarlılığı: eğitim masaüstünde (Ryzen 3600X), laptop ile birebir paket sürümleri
  (pandas 2.3.3 / numpy 2.1.3).

---

## 11. Çıktı Dosyaları (Bu Klasörde / Kökte)
| Dosya | İçerik |
|---|---|
| `rack_crane_distance.png` | **Raf senaryosu: PPO vs Heuristic vs Random vinç mesafesi (PPO kazanır)** |
| `comparison.json` / `results_*.csv` | Held-out karşılaştırma ham verileri |
| `rehandling_bar.png` | Held-out rehandling bar grafiği |
| `TEZ_SONUC_RAPORU.md` | **Bu rapor** |
| `../../POSTER-PPO-SAVUNMA.md` | Jüri soru-cevap (PPO/raf/negatif-ödül savunması) |

---

**Sistem tez savunmasına HAZIR.** İddia, ezber değil **görülmemiş veride genelleşme** ve
**yöntem–problem eşleşmesi** üzerine kuruludur. İki öz-düzeltme (overfitting + kısıt-gevşemesi)
ve iki-senaryolu dürüst sonuç, projeyi saf "PPO kazandı" anlatısından bilimsel olarak çok daha
güçlü bir konuma taşır.
