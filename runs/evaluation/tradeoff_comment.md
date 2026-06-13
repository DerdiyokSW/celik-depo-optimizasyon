**Çok-amaçlı tradeoff yorumu** (kaynak: comparison.json, gerçek koşum):

- En düşük **rehandling**: **PPO** (3.40) — pahalı sevkiyat operasyonunu (üst bobini kaldırma) minimize eder.
- En düşük **vinç mesafesi**: **Heuristic** (27081 m) — toplam vinç hareketini (mesafe-açgözlü) minimize eder.

Bu iki hedefin farklı politikalarda minimize olması, problemin **çok-amaçlı** doğasını gösterir: rehandling ve toplam vinç mesafesi kısmen çelişen hedeflerdir. PPO, az-rehandling için bazen daha uzak ama 'temiz' (engelsiz) konumlar seçerek mesafeyi artırır; Heuristic ise her hamlede en yakın konumu seçtiğinden mesafeyi düşürür ama acil bobinleri bazen gömerek rehandling'i artırır. Operasyonel öncelik (vinç enerjisi mi, sevkiyat hızı mı) hangi politikanın tercih edileceğini belirler.
