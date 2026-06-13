"""Aşama 6 — Pekiştirmeli öğrenme (PPO) paketi.

Projenin yıldızı ve tek gerçek öğrenen bileşeni. Simülasyon çekirdeğini bir
gymnasium ortamı olarak sarar (``WarehouseEnv``) ve MaskablePPO ile bobin
yerleştirmeyi deneme-yanılmayla öğrenen bir ajan eğitir.

Not: Bu ``__init__`` bilinçli olarak HİÇBİR şey içe aktarmaz. ``warehouse_env`` ve
``train_ppo`` ağır bağımlılıklara (gymnasium, torch, sb3) dayanır; bunları paket
seviyesinde yüklemek, yalnızca hafif ``action_space``/``observation`` modüllerine
ihtiyaç duyan kodu (ör. ``PPOPolicy``) gereksiz yere ağırlaştırır.
"""
