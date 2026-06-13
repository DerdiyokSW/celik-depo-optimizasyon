# 09 — Savunma Notları (Jüri Soru-Cevap Rehberi)

> Bu dosya **kod üretmez**. Tez savunmasında jürinin sorma olasılığı yüksek
> sorulara hazırlık rehberidir. Her cevap, projenin gerçek tasarım kararlarına
> dayanır; ezber değil, sistemi anlamanın doğal sonucudur.

---

## 1. "Bu gerçekten bir makine öğrenmesi problemi mi? Temelde optimizasyon değil mi?"

Dürüst ve güçlü cevap: **Evet, temelde bir Yöneylem Araştırması / kombinatoryal
optimizasyon problemidir.** Her şey kesin (deterministik) olsaydı iyi bir
sezgisel algoritma yeterdi. Ama gerçek dünya stokastiktir: araç gecikir, sipariş
iptal olur, üretim yığılır. Makine öğrenmesi tam bu belirsizliği yönetmek için
devrededir — (i) tahminleyici ML belirsizliği ölçülebilir sinyale çevirir,
(ii) PPO bu belirsizlik altında adaptif karar vermeyi öğrenir. Yani problem OR
problemidir; ama *adaptif ve ölçeklenebilir* çözümü ML gerektirir. Bu ayrımı
net yapmak jüride profesyonel durur.

## 2. "Sistemde 'dinamik öğrenme' tam olarak nerede?"

En kritik soru. Üç katmanı ayırarak anlat (`00 §4`):
- **Katman 1 — Yeniden planlama:** Olay gelince plan yeniden hesaplanır. Dinamik
  davranıştır, öğrenme değildir.
- **Katman 2 — Genelleyen PPO politikası:** Ajan, gecikme/iptal/peak olaylarının
  bol yaşandığı geniş bir dağılımda eğitilir; görmediği durumlara genellemeyle
  tepki verir. "Öğrenilmiş adaptasyon" budur.
- **Katman 3 — Periyodik yeniden eğitim:** Biriken yeni veriyle model warm-start
  ile yeniden eğitilir. `retraining_experiment.py`'nin ürettiği aşağı yönlü eğri
  bunun görsel kanıtıdır.
Tuzak: "Sistem canlı öğreniyor" deme. Canlı online RL yapılmadı; çünkü lisans
düzeyinde riskli ve kararsız. Katman 2 + 3 zaten "dinamik öğrenme"yi karşılar.

## 3. "Veri sentetik. Gerçek veriyle çalışmadan sonuçlar anlamlı mı?"

Cevap: Gerçek endüstriyel veriye erişim yoktu; bu, alanın bilinen bir kısıtıdır.
Ama veri rastgele değil — fabrika kataloglarındaki gerçek fiziksel sınırlar
(ağırlık, çap aralıkları) ve gerçek dünya davranışları (kötü havada gecikme
artışı, düşük sicilli firmanın daha çok gecikmesi) bilinçle gömüldü. Veri
tutarlılığı 8 doğrulama kuralıyla otomatik denetlendi. Sistemin tasarımı veriden
bağımsızdır: yarın gerçek veri gelse aynı hat çalışır. Sınırı dürüstçe kabul et;
bu zayıflık değil, bilimsel dürüstlüktür.

## 4. "Tahmin modelinde veri sızıntısı (data leakage) var mı?"

Hayır. Model yalnızca araç yola çıkmadan **bilinebilecek** özellikleri görür:
hava durumu, firma sicili, mesafe, planlanan saat/gün/ay, araç tipi, trafik.
`actual_arrival` ve `delay_minutes` (hedefin kendisi) özellik matrisine asla
girmez; bu bir testle (`test_no_leakage`) garanti altına alınmıştır.

## 5. "Neden Q-Learning değil de PPO?"

Depo durum uzayı kombinatoryal olarak devasadır. Q-Learning her durum için tablo
tutar; bu uzayda bellek çöker. PPO derin pekiştirmeli bir algoritmadır,
politikayı sinir ağıyla temsil eder, tablo tutmaz. Ayrıca PPO'nun "proximal"
(clipped) hedefi, bir güncellemede politikanın aşırı sapmasını engeller —
öğrenme kararlıdır. Bu iki sebep, PPO'yu bu problem için doğru seçim yapar.

## 6. "Performans sayıları (rehandling, MAE) nasıl elde edildi? Güvenilir mi?"

Tüm sayılar gerçekten çalıştırılan kodun çıktısıdır; hiçbiri elle girilmedi.
Karşılaştırma, dört politikanın **birebir aynı** tohumlanmış senaryolarda
koşturulmasıyla yapıldı; fark Wilcoxon signed-rank testiyle istatistiksel olarak
doğrulandı. (Not: Erken durum raporlarındaki sayılar göstermeliktir ve referans
alınmadı — gerçek hat onların yerine geçti.)

## 7. "Gecikme modelinin R²'si neden 1.0 değil?"

Çünkü veri üreticisi gecikmeye, gerçek dünyadaki öngörülemezliği temsil eden,
standart sapması ~8 dakika olan indirgenemez bir gürültü ekler. Bu, hata için
teorik bir alt sınır koyar — *hiçbir model* bunun altına inemez. R²'nin 1.0
olmaması bir kusur değil, gerçekçiliğin doğal sonucudur.

## 8. "PPO, sezgisel yöntemi geçemezse proje başarısız mı?"

Hayır. PPO'nun ML-destekli sezgiseli geçememesi de **geçerli bir bilimsel
sonuçtur** — "bu problem ve bu ölçekte, iyi tasarlanmış bir sezgisel rekabetçi
kalır" bulgusu da bir katkıdır. `PlacementPolicy` ortak arayüzü sayesinde sistem
PPO olmadan da uçtan uca çalışır. Önemli olan karşılaştırmanın dürüst ve
yöntemsel olarak sağlam olmasıdır; sonucun yönü değil.

## 9. "İstif modeliniz gerçek piramit fiziğini yansıtıyor mu?"

Kısmen. Gerçek piramit fiziği bay ekseni boyunca komşuluk gerektirir. Bu proje,
bilinçli bir soyutlama kullanır: her `(zone, bay)` hücresi 3 katlık bir dikey
sütundur ve piramit fiziği iki kuralla temsil edilir — üst kat için alt kat dolu
olmalı, yukarı çıkıldıkça ağırlık azalmalı. Bu, lisans düzeyi coil-stacking
modellerinde standart ve savunulabilir bir basitleştirmedir. Gerçek çapraz-bay
piramidi, açıkça tanımlanmış bir gelecek iyileştirmesidir (`01 §4`).

## 10. "Neden saf MILP ile kesin çözüm bulmuyorsunuz?"

Problem NP-Hard'dır; MILP değişken sayısı arttıkça (gerçek depo ölçeğinde
binlerce bobin) makul sürede çözüm üretemez. MILP yalnızca **küçük bir örnekte**
bir kıyaslama (baseline) aracı olarak kullanılabilir — tüm depoya ölçeklenemez.
Sistemin değeri, anlık (milisaniyeler içinde) ve ölçeklenebilir karar
üretmesidir.

## 11. "Action masking nedir, neden ceza vermek yerine onu kullandınız?"

Geçersiz yerleştirmeleri (dolu slot, fizik ihlali) cezalandırmak, ajanın öğrenme
kapasitesinin büyük kısmını "yasak hamle yapmamayı" öğrenmeye harcamasına yol
açar. Action masking ile geçersiz eylemler ajana hiç sunulmaz; ajan yalnızca
geçerli konumlar arasından seçer ve tüm öğrenme kapasitesini gerçek hedefe
(rehandling azaltma) ayırır. `sb3-contrib`'in MaskablePPO'su bunu sağlar.

## 12. "Ödül fonksiyonunu nasıl tasarladınız?"

Üç bileşen: (i) küçük bir yönlendirme ödülü — erken öğrenmeyi hızlandıran, vekil
bir yerleştirme-kalitesi sinyali; (ii) gerçekleşen sevkiyat sinyali — asıl hedef
olan gerçek rehandling cezası; (iii) bölüm sonu terminal ödülü — toplam
performansın bir referansla kıyası. Bir yerleştirmenin yol açtığı rehandling çok
sonra gerçekleşir; bu gecikmiş kredi atamasını PPO'nun değer fonksiyonu ve GAE'si
çözer. Geçersiz hamle cezası yoktur — o iş masking'e aittir.

## 13. Demo önerisi (jüriye canlı gösterim)

1. Dashboard'u aç, Klasik Sezgisel ile otonom akışı başlat — depo dolarken
   metrikleri göster.
2. "Olay Tetikle (Peak)" ile kriz senaryosu başlat — sistemin tepkisini göster.
3. Politikayı PPO'ya çevir, aynı senaryoyu koştur — rehandling farkını göster.
4. Değerlendirme grafiklerini aç: dört politikanın karşılaştırması, dayanıklılık
   eğrileri ve yeniden eğitim eğrisi.

## 14. Sistemi anlatma sırası (mimari gezintisi)

Veri Üretici → Simülasyon Çekirdeği (+ Event Generator) → Gecikme Tahmin ML →
Yerleşim Politikaları (ortak `PlacementPolicy` arayüzü) → Dashboard → PPO →
Değerlendirme. Vurgu: dört politikanın aynı arayüzü paylaşması (stabilite ve
adil karşılaştırma), ve hibrit yapının kalbi — ML'in tahmininin hem ML-destekli
sezgisele hem PPO gözlemine girdi olması.

## 15. Zayıf noktaları önceden kabul et

İyi bir savunma, sınırları jüriden önce kabul eder: veri sentetiktir; istif
modeli bir soyutlamadır; gerçek online öğrenme yapılmadı; MILP yalnızca küçük
örnekte çalışır. Bunları zayıflık olarak değil, **bilinçli kapsam kararları ve
gelecek çalışma yönleri** olarak sunmak, projeyi daha güçlü gösterir.
