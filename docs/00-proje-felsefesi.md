# 00 — Proje Felsefesi ve Problem Tanımı

> Bu dosya kod üretmez. Sistemin *neden* böyle tasarlandığını anlatır. Claude Code
> bu dosyayı okuduğunda projenin amacını ve kavramlarını tam kavramış olmalıdır.
> Geliştirici için bu dosya aynı zamanda tez savunmasının kavramsal temelidir.

---

## 1. Problem nedir?

Ağır sanayi çelik tesislerinde üretim bandından çıkan çelik bobinler (rulolar)
bir depoya alınır, sevkiyat zamanı gelince TIR, tren veya gemiye yüklenir.
Bobinler 10–30 ton ağırlığında, silindirik ve hacimli ürünlerdir. Depoda yerden
tasarruf için **piramit (saddle) istif** yöntemiyle üst üste dizilirler: alt
sırada birbirine bitişik iki bobinin oluşturduğu "yuva"ya bir üst bobin oturur.
Güvenlik nedeniyle genellikle en fazla 3 kat çıkılır.

Bu istif yöntemi alanı verimli kullanır ama bir bedel doğurur: **alttaki bir
bobine ulaşmak için üstündeki bobinleri önce başka yere kaldırmak gerekir.** Bu
fazladan vinç hamlesine **rehandling** denir. Rehandling; vinç operatörü zamanı,
enerji, üretim bandı tıkanması ve gecikme cezaları (demoraj) açısından gerçek ve
ciddi bir maliyettir.

**Projenin çözmeye çalıştığı çekirdek soru:** Üretimden çıkan her bobini depoya
*ilk seferde* öyle bir konuma yerleştirelim ki, sevkiyat günü geldiğinde
rehandling sayısı minimum olsun.

İdeal kural sezgisel olarak basittir — "erken sevk edilecek bobini erişilebilir
(üst/ön) konuma, geç sevk edilecek bobini alta koy". Problemi zorlaştıran şey bu
kuralı **belirsizlik altında ve devasa ölçekte** uygulamaktır (aşağıdaki 2 ve 3).

## 2. Bu problem neden zor? (NP-Hard ve kombinatoryal patlama)

Problem, literatürde **Storage Location Assignment Problem (SLAP)** ailesine,
özelde **Coil Stacking Problem** ve **Crane Scheduling** kesişimine girer.

10 bobini bir alana kaç farklı sırayla dizebileceğimiz `10!` ≈ 3.6 milyondur.
100 bobinde olasılık uzayı `100!` — pratikte tüketilemez. Bu yüzden problem
**NP-Hard**'dır: kaba kuvvet veya saf if-else algoritmaları ölçek büyüdükçe
çöker. Saf MILP (kesin matematiksel optimizasyon) yaklaşımı da küçük örneklerde
(~11 konum üstü) makul sürede çözüm üretemez.

## 3. Bu problem neden makine öğrenmesi gerektiriyor?

Eğer her şey kesin (deterministik) olsaydı — araçların ne zaman geleceği belli,
arıza yok — iyi bir sezgisel algoritma yeterdi. Ama gerçek dünya **stokastiktir**:

- Bir TIR planlanan 14:00 yerine 17:00'de gelebilir (hava, trafik, firma sicili).
- Bir sipariş iptal olabilir, acil bir sipariş araya girebilir.
- Gemi erken yanaşabilir, üretimde yığılma olabilir.

Klasik bir algoritma bu her değişimde tüm planı baştan hesaplamak zorundadır.
Makine öğrenmesi — özellikle pekiştirmeli öğrenme — bu noktada parlar: önceden
çok sayıda senaryoda eğitildiği için, daha önce birebir görmediği durumlara da
**genelleme** ile anında tepki verebilir.

Bu projede ML iki yerde devrededir:

1. **Tahminleyici ML (gözetimli):** Aracın gerçek varış gecikmesini tahmin eder.
   Yani belirsizliğin *bir kısmını ölçülebilir sinyale çevirir*.
2. **PPO (pekiştirmeli):** Bu tahmini ve deponun anlık durumunu girdi alıp
   bobinin nereye konacağını öğrenir.

## 4. "Dinamik öğrenme" tam olarak ne demek? (Savunma için kritik)

Bu projede en çok karıştırılan ve jürinin en çok soracağı konu budur. "Sistem
öğrenir ve adapte olur" cümlesi **üç ayrı mekanizmayı** kapsar. Geliştirici
bunları ayırarak anlatabilmelidir:

**Katman 1 — Tepkisel yeniden planlama (re-planning).**
Bir olay geldiğinde (yeni sipariş, gecikme, iptal) yerleşim planının ilgili
kısmı yeniden *hesaplanır*. Bunu hem sezgisel motor hem PPO yapar. Bu dinamik
bir *davranıştır* ama **öğrenme değildir** — sadece tepkidir.

**Katman 2 — Genelleyen politika (PPO çekirdeği).**
PPO ajanı; gecikme, iptal ve yoğun (peak) olayların *bol bol* yaşandığı bir
ortamda milyonlarca adım eğitilir. Sonuçta öğrendiği politika, daha önce birebir
görmediği durumlara da iyi tepki verir. "Öğrenilmiş adaptasyon" budur. Ajan,
canlı çıkarım anında ağırlıklarını güncellemez; ama politikası state'in
fonksiyonu olduğu için davranışı duruma göre değişir.

**Katman 3 — Periyodik yeniden eğitim (asıl "dinamik ML").**
Simülasyon ilerledikçe yeni araç kayıtları birikir. Her simüle gün/hafta sonunda
gecikme tahmin modeli (ve istenirse PPO) biriken veriyle **warm-start ile
yeniden eğitilir**. Bu offline/batch bir işlemdir — bu yüzden güvenli ve
stabildir. Jüriye gösterilecek görsel kanıt da budur: yeniden eğitim döngüleri
boyunca aşağı inen bir performans eğrisi.

**Yapılmayacak olan:** Gerçek online/continual RL (canlı ağırlık güncelleme).
Lisans bitirmesi için riskli ve stabilitesi düşüktür. Katman 2 + 3 jüriyi
fazlasıyla tatmin eder.

Özet: Sistem "adaptif"tir çünkü (Katman 1) her olaya yeniden plan yapar,
(Katman 2) geniş dağılımda eğitilmiş genelleyen bir politikaya sahiptir ve
(Katman 3) zamanla biriken veriyle kendini yeniden eğitebilir.

## 5. Dört yaklaşımın karşılaştırılması (projenin bilimsel iddiası)

Projenin tezi şudur: *Gecikme tahminini yerleşim kararına dahil etmek ve
öğrenen bir ajan kullanmak, klasik yöntemlere göre rehandling'i belirgin
biçimde düşürür.* Bunu kanıtlamak için aynı simülasyon senaryolarında dört
politika karşılaştırılır:

| Politika | Açıklama | Beklenen rol |
|---|---|---|
| Rastgele | Bobini geçerli ama rastgele bir konuma koyar | Alt sınır (baseline) |
| Klasik sezgisel | Sevkiyat zamanına göre skorlar, kuralla yerleştirir | Güçlü referans |
| ML-destekli sezgisel | Aynı sezgisel + gecikme tahmini girdisi | ML'in net katkısını gösterir |
| PPO | Öğrenen ajan | Yüksek olay yoğunluğunda fark yaratması beklenir |

Karşılaştırma metrikleri: rehandling sayısı, toplam vinç hareket mesafesi,
yükleme süresi, depo doluluk oranı, karar hesaplama süresi ve metriklerin
standart sapması (kararlılık). Detayları `08-asama-7-degerlendirme.md` içinde.

## 6. Akademik referanslar ve onlardan alınanlar

İki literatür çalışması, *ana modeli değiştirmeden* ortamı zenginleştirmek için
tersine mühendislikle kullanılır:

- **J4n1k (MILP / affinity tabanlı yaklaşım):** "Birlikte sipariş edilen ürünler
  yakına konur" mantığını sunar. Bu projede şuna **uyarlanır**: *aynı araca/
  lojistik hattına gidecek bobinler aynı zone'a konur* (affinity ödülü). Ayrıca
  depo içi mesafe için Manhattan mesafesi formülasyonu buradan alınır. MILP'in
  kendisi opsiyonel bir "kusursuz çözüm" baseline'ı olarak kullanılabilir.
- **Knapp (Deep Learning / SBS-RS):** Sistemdeki ani dalgalanmalara (peak
  situations) odaklanır. Bu projede *peak olay senaryosu* (acil sipariş dalgası,
  üretim yığılması) olarak event generator'a uyarlanır.

Not: Bu projenin yerleştirme felsefesi affinity *değil*, **zaman-öncelikli ve
katmanlı** yerleştirmedir — beklenen sevkiyat zamanı yakın bobinler erişilebilir
konuma, uzak olanlar alt katmana. Affinity yalnızca ikincil bir ödül bileşenidir.

## 7. Sözlük

- **Rehandling:** Hedeflenen bobine ulaşmak için üstündeki engelleyici bobini
  başka yere kaldırma — gereksiz vinç hamlesi. Sistemin minimize ettiği ana maliyet.
- **Piramit / saddle istif:** Silindirik bobinlerin alt sıradaki iki bobin
  arasına oturacak şekilde üst üste dizilmesi.
- **SLAP (Storage Location Assignment Problem):** Hangi ürünün depoda nereye
  konacağı problemi. Bu projenin ait olduğu literatür ailesi.
- **Coil Stacking Problem:** SLAP'ın çelik bobinlere özgü, istif kısıtlı hali.
- **Zone / Bay / Stack:** Depo koordinat sistemi. Zone = bölge (lojistik
  hattına göre gruplanır), Bay = bölge içindeki sıra, Stack = dikey istif kolonu.
  Stack içindeki dikey konum **layer** (kat) ile ifade edilir.
- **ETA (Estimated Time of Arrival):** Aracın planlanan varış zamanı.
- **Gecikme (delay):** Gerçek varış ile planlanan varış arasındaki fark (dakika).
  Tahminleyici ML modelinin hedef değişkeni.
- **Baseline:** Karşılaştırma için referans yöntem (ör. rastgele yerleşim).
- **MDP (Markov Decision Process):** Pekiştirmeli öğrenmenin matematiksel
  çerçevesi: durum (state), eylem (action), ödül (reward) üçlüsü.
- **Observation / State:** Ajanın o an gördüğü bilgi (depo doluluk haritası +
  bekleyen bobin + gecikme tahmini).
- **Action:** Ajanın seçtiği karar (bobinin yerleştirileceği konum).
- **Reward:** Eylemin iyi/kötü olduğunu söyleyen sayısal sinyal.
- **Policy (politika):** Durumdan eyleme eşleme — ajanın "stratejisi".
- **Action masking:** Geçersiz eylemleri (dolu slot, fizik ihlali) ajana hiç
  sunmama tekniği. Cezalandırmaktan daha verimlidir; ajan kapasitesini gerçek
  hedefe ayırır. `sb3-contrib` MaskablePPO ile uygulanır.
- **PPO (Proximal Policy Optimization):** Modern, kararlı bir derin pekiştirmeli
  öğrenme algoritması. Politikayı bir sinir ağıyla temsil eder; Q-Learning'in
  aksine tablo tutmaz, devasa durum uzaylarında çöker.
- **Heuristic (sezgisel):** Optimumu garanti etmeyen ama hızlı ve açıklanabilir
  kural tabanlı algoritma.
- **Event-driven simülasyon:** Zamanın olaylarla (yeni sipariş, gecikme...)
  ilerlediği simülasyon modeli.
- **Demoraj:** Araç/geminin zamanında yüklenememesi nedeniyle kesilen bekleme
  cezası — rehandling'in dolaylı gerçek dünya maliyeti.
