# 08 — Aşama 7: Değerlendirme ve Karşılaştırmalı Analiz

> **Önkoşul okuma:** Tüm önceki docs (`00`–`07`).
>
> Bu aşama projenin **bilimsel iddiasını kanıtladığı** yerdir: dört yaklaşım
> aynı koşullarda kıyaslanır ve hangisinin ne zaman üstün olduğu ortaya konur.

---

## 1. Amaç

Dört yerleşim politikasını (Rastgele, Klasik Sezgisel, ML-destekli Sezgisel,
PPO) aynı simülasyon senaryoları üzerinde sistematik olarak karşılaştırmak;
sonuçları istatistiksel olarak doğrulamak; sistemin "dinamik öğrenme"
kabiliyetini deneyle göstermek.

## 2. Üretilecek dosyalar (kod)

```
src/evaluation/
├── __init__.py
├── scenario.py             # tohumlanmış senaryo üretimi (tüm politikalar aynısını görür)
├── runner.py               # bir politikayı N senaryoda koşturup metrik toplama
├── compare.py              # 4 politikayı karşılaştır + istatistiksel test (main)
├── robustness.py           # olay hızını değiştirerek dayanıklılık analizi
├── retraining_experiment.py# periyodik yeniden eğitim deneyi (dinamik ML kanıtı)
└── milp_baseline.py        # OPSİYONEL — PuLP ile küçük örnek optimal çözüm
```

Grafikler ve tablolar `runs/evaluation/` altına yazılır.

## 3. Adil karşılaştırmanın temeli: tohumlanmış senaryolar (`scenario.py`)

Karşılaştırmanın geçerli olması için dört politikanın da **birebir aynı**
senaryoları görmesi şarttır.

```python
def make_scenarios(n: int, base_seed: int) -> list[ScenarioSpec]:
    """N adet tekrarlanabilir senaryo üretir. Her senaryo; başlangıç durumu,
    bobin akışı, araç varışları ve olay akışı için sabit tohumlar içerir.
    Aynı base_seed her zaman aynı senaryo kümesini verir."""
```

Bir `ScenarioSpec`, simülatörü ve event generator'ı deterministik kurmak için
gereken tüm tohumları ve parametreleri taşır.

## 4. Koşum ve metrik toplama (`runner.py`)

```python
def evaluate_policy(policy: PlacementPolicy,
                    scenarios: list[ScenarioSpec]) -> pd.DataFrame:
    """Bir politikayı tüm senaryolarda koşturur; her senaryo için
    SimulationMetrics toplar. Dönüş: senaryo başına bir satır içeren DataFrame.
    """
```

Toplanan metrikler (her senaryo için): rehandling sayısı, toplam vinç hareket
mesafesi (m), ortalama yükleme süresi (dk), depo doluluk oranı, bobin başına
karar hesaplama süresi (ms).

## 5. Karşılaştırma ve istatistiksel doğrulama (`compare.py`)

1. Dört politika da aynı senaryo kümesinde koşturulur (varsayılan **N = 100+**
   senaryo — istatistiksel anlamlılık için).
2. Her metrik için ortalama ± standart sapma raporlanır. Standart sapma ayrıca
   bir **kararlılık** ölçüsüdür (düşük sapma = öngörülebilir politika).
3. **İstatistiksel test:** Aynı senaryolar kullanıldığı için eşleştirilmiş
   (paired) karşılaştırma yapılır. Dağılım varsayımı gerektirmeyen
   **Wilcoxon signed-rank testi** kullanılır. İki politika arasındaki farkın
   anlamlı olup olmadığı p-değeri ile raporlanır; ayrıca etki büyüklüğü verilir.
4. Çıktı: karşılaştırma tablosu + bar grafikleri (`runs/evaluation/`).

**Beklenen bulgu (iddia):** Rehandling açısından `Rastgele > Klasik Sezgisel >
ML-destekli Sezgisel`, ve yeterince eğitilmiş PPO'nun ML-destekli sezgisele
yakın ya da onu geçen sonuç vermesi. Gerçek sayılar bu hattın çıktısıdır;
hiçbir değer önceden varsayılmaz veya uydurulmaz.

## 6. Dayanıklılık analizi (`robustness.py`)

Tek bir olay hızında karşılaştırma yetmez; sistemin **stres altındaki**
davranışı ölçülür. Olay hızı parametre olarak değiştirilir (8, 12, 20 olay/saat)
ve dört politika her hızda yeniden değerlendirilir.

Beklenti: Klasik sezgisel düşük yükte hızlı ve yeterlidir; olay yoğunluğu
arttıkça PPO'nun farkı açması beklenir. Çıktı: olay hızına karşı rehandling
eğrileri (`runs/evaluation/robustness.png`).

## 7. Dinamik öğrenme deneyi (`retraining_experiment.py`)

Bu deney, `00-proje-felsefesi.md` §4 **Katman 3**'ün somut kanıtıdır ve
savunmada "sistem zamanla öğreniyor" iddiasının görsel dayanağıdır.

Kurgu:
1. Simülasyon zaman içinde ilerletilir; yeni araç kayıtları birikir.
2. Belirli aralıklarla (ör. her simüle hafta) gecikme tahmin modeli — ve
   istenirse PPO, `--resume-from` ile warm-start — biriken veriyle **yeniden
   eğitilir**.
3. Her yeniden eğitim döngüsünden sonra sistemin rehandling performansı ölçülür.
4. Çıktı: yeniden eğitim döngülerine karşı rehandling eğrisi — **aşağı yönlü**
   bir eğri, sistemin biriken veriyle kendini iyileştirdiğini gösterir
   (`runs/evaluation/retraining_curve.png`).

## 8. MILP baseline (`milp_baseline.py`) — OPSİYONEL

`00-proje-felsefesi.md` §6: J4n1k esinli kesin matematiksel kıyaslama. **Bu
modül gerçekten opsiyoneldir** — zaman daralırsa kesilir, projenin geçerliliği
buna bağlı değildir.

Kurgu: `PuLP` ile, depo ve sipariş uzayının **küçük bir alt kümesi** (ör. 20
bobin, 5×5 alan) için rehandling/mesafeyi minimize eden kesin çözüm hesaplanır.
Aynı küçük örnek dört politikayla da çözülür; sonuçlar "optimallik açığı" olarak
raporlanır (ör. "PPO, optimal çözümün %X'ine ulaşıyor").

Uyarı: MILP NP-Hard olduğu için yalnızca küçük örnekte çalışır; tüm depoya
ölçeklenemez — bu sınır savunmada açıkça belirtilir.

## 9. Nihai çıktı: karar tablosu

`compare.py` ve `robustness.py` sonuçları birleştirilerek, hangi yaklaşımın
hangi koşulda öne çıktığını gösteren açık bir **karar tablosu** üretilir.
Örnek biçim (değerler gerçek koşumdan gelir):

| Koşul | Önerilen yaklaşım | Gerekçe |
|---|---|---|
| Düşük olay yoğunluğu | Klasik / ML sezgisel | Hızlı, yeterli, açıklanabilir |
| Yüksek olay yoğunluğu | PPO | Stres altında daha iyi genelleme |
| Açıklanabilirlik kritik | ML-destekli sezgisel | Karar gerekçesi şeffaf |

## 10. Kabul kriterleri

1. `python -m src.evaluation.compare` dört politikayı N senaryoda koşturur;
   tablo + grafik üretir.
2. Karşılaştırma eşleştirilmiş senaryolar üzerinde yapılır; Wilcoxon p-değerleri
   raporlanır.
3. `robustness.py` üç olay hızında eğrileri üretir.
4. `retraining_experiment.py` yeniden eğitim eğrisini üretir ve eğri öğrenmeyi
   (aşağı yönlü eğilim) gösterir.
5. Tüm sayılar gerçek koşum çıktısıdır; hiçbiri elle girilmemiştir.
6. (Opsiyonel) `milp_baseline.py` küçük örnekte optimallik açığını raporlar.

## 11. Testler (`tests/test_evaluation/`)

- `test_scenarios_reproducible`: Aynı `base_seed` aynı senaryo kümesini verir.
- `test_same_scenarios_all_policies`: Dört politika da birebir aynı senaryoları
  görür (adil karşılaştırma garantisi).
- `test_metrics_collected`: `evaluate_policy` her senaryo için tüm metrikleri
  doldurur.
- `test_stat_test_runs`: Wilcoxon testi geçerli p-değeri üretir.

## 12. Bu aşama bittiğinde elde olan

Projenin bilimsel sonucu: dört yaklaşımın istatistiksel olarak doğrulanmış
karşılaştırması, dayanıklılık eğrileri, dinamik öğrenme kanıtı ve bir karar
tablosu. Sistem artık uçtan uca tamamlanmış ve savunmaya hazırdır.
