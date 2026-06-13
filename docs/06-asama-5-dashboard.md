# 06 — Aşama 5: Simülasyon Dashboard'u (Plotly + Dash)

> **Önkoşul okuma:** `00`, `01`, `03-asama-2-simulasyon-cekirdegi.md`,
> `05-asama-4-yerlesim-politikalari.md`.
>
> Bu aşama bir **görselleştirme katmanıdır** — simülasyon çekirdeğinin üstüne
> oturan ince bir kabuk. Çekirdeğe veya politikalara hiçbir mantık eklemez,
> yalnızca onları görünür kılar. PPO eğitimi bu katmandan tamamen bağımsızdır
> (eğitim arayüzsüz/headless yapılır; dashboard yalnızca *sonucu* izletir).

---

## 1. Amaç

Üretilen veriyi, depo durumunu ve yerleşim politikalarının kararlarını canlı
olarak gösteren bir masaüstü/tarayıcı dashboard'u kurmak — bir "Dijital İkiz".
Jüri sistemi burada izler: 3B depo, akan metrikler, vinç hareketleri ve
"olay tetikleme" ile sistemin tepkisi.

## 2. Neden Plotly + Dash

Dash, Plotly üzerine kurulu, tarayıcıda çalışan bir Python dashboard çatısıdır.
3B görselleştirme (`Scatter3d`) Plotly'de hazır gelir; PyQt'de 3B ızgara
çizdirmeye kıyasla çok daha az iş. Simülasyon çekirdeği sunucu tarafında çalışır,
Dash yalnızca durumu çizer. `dcc.Interval` ile otomatik oynatma sağlanır.

## 3. Üretilecek dosyalar (kod)

```
src/dashboard/
├── __init__.py
├── app.py              # Dash uygulaması — layout + callback kayıtları (main)
├── warehouse_view.py   # WarehouseState -> 3B Plotly figürü
├── panels.py           # metrik panelleri ve log konsolu bileşenleri
└── controllers.py      # buton / interval callback mantığı (sim ile köprü)
```

## 4. Arayüz bileşenleri

**4.1. 3B Depo Görünümü (`warehouse_view.py`)**
`WarehouseState.snapshot()` çıktısını bir `plotly.graph_objects.Scatter3d`
figürüne çevirir:
- Eksenler: X = bay, Y = zone, Z = layer (kat).
- Dolu konumlar: renk = bobinin `urgency_score`'u (renk skalası; acil = sıcak
  renk, acil değil = soğuk renk).
- Boş/rezerve konumlar: koyu renkli işaretler.
- Üzerine gelince (hover): bobin ID, tip, ağırlık, hedef lojistik hattı.

```python
def render_warehouse(state: WarehouseState, layout: WarehouseLayout) -> go.Figure:
    """Anlık depo durumundan 3B Plotly figürü üretir.
    Renk = sevkiyat aciliyeti; koyu işaretler boş konumları gösterir."""
```

**4.2. Canlı Metrik Paneli (`panels.py`)**
Anlık olarak gösterilen değerler: depo doluluk oranı (%), toplam rehandling
sayısı, ortalama yükleme süresi, çalışan politikanın adı, simülasyon saati,
(varsa) gecikme modelinin test MAE'si.

**4.3. Log Konsolu (`panels.py`)**
Olan biteni satır satır akıtan bir metin alanı: yeni bobin girişleri, vinç
hareketleri, gerçekleşen olaylar (gecikme, iptal), rehandling uyarıları.

**4.4. Kontroller (`controllers.py`)**
- **Politika Seçici:** Random / Heuristic / MLHeuristic / PPO arasından seçim
  (PPO yalnızca eğitilmiş model varsa aktif).
- **Manuel Adım İlerle:** Tek bir yerleştirme adımı koşar.
- **Otonom Akış Başlat/Durdur:** `dcc.Interval` ile sürekli adım akışı.
- **Olay Tetikle (Peak):** `EventGenerator.trigger_peak()` çağırır; sistemin
  kriz tepkisini canlı izletir.

## 5. Çalışma mantığı

Dashboard bir `WarehouseSimulator` örneğini sunucu tarafında tutar. Seçilen
politika simülatöre takılır. "Adım" tetiklendiğinde `pending_coil` →
`policy.decide` → `apply_placement` ilkelleri çağrılır, sonuç figüre ve panellere
yansıtılır. Yani dashboard, Aşama 4'teki değerlendirme döngüsünün **görsel ve
adım adım** halidir — yeni bir mantık değil.

## 6. İş kuralları

- Dashboard simülasyon çekirdeğini yalnızca **okur ve adımlatır**; fizik, metrik
  veya karar mantığı içermez.
- PPO eğitimi burada yapılmaz. Eğitilmiş bir PPO modeli varsa `PPOPolicy` ile
  diğer politikalar gibi takılıp izletilebilir.
- Otomatik oynatma hızı (`dcc.Interval` aralığı) kullanıcı tarafından
  ayarlanabilir; simülasyon hızı görselleştirmeden bağımsızdır.

## 7. Kabul kriterleri

1. `python -m src.dashboard.app` Dash sunucusunu başlatır, tarayıcıda açılır.
2. 3B depo görünümü gerçek `WarehouseState`'i doğru yansıtır (dolu/boş konumlar,
   aciliyet renkleri).
3. Politika seçici çalışır; üç baseline politika canlı koşturulabilir.
4. "Otonom Akış" sürekli adım akıtır; metrikler ve log canlı güncellenir.
5. "Olay Tetikle" peak senaryosunu başlatır ve etkisi görünür.
6. Dashboard kapatıldığında simülasyon çekirdeği veya veri bozulmaz.

## 8. Testler (`tests/test_dashboard/`)

Arayüz testleri hafiftir; ağırlık figür/bileşen üreticilerindedir.
- `test_render_warehouse_valid`: `render_warehouse` geçerli bir `go.Figure`
  döndürür; nokta sayısı dolu konum sayısıyla tutarlı.
- `test_color_mapping`: Yüksek aciliyetli bobin ile düşük aciliyetli bobin
  farklı renge eşlenir.
- `test_controller_step`: Adım callback'i simülatörü bir adım ilerletir.

## 9. Bu aşama bittiğinde elde olan

Çalışan, izlenebilir bir Dijital İkiz dashboard'u. Sistem artık jüriye
gösterilebilir durumdadır. Sıradaki aşama (6) öğrenen PPO ajanını ekler.
