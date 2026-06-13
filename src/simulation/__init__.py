"""Aşama 2 — Arayüzsüz (headless), deterministik depo simülasyon çekirdeği.

Bu paket projenin en kritik mimari katmanıdır: depo 3B durumunu tutar, fizik/istif
kısıtlarını uygular, geçerli yerleştirme konumlarını sorgular (Aşama 6'daki
MaskablePPO action mask'inin temeli), sevkiyatta rehandling sayar ve zamanı
olaylarla ilerletir. Hem politika-güdümlü değerlendirme hem dışarıdan-güdümlü RL
modunda aynı ilkelleri kullanır.
"""
