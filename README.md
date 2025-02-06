# Stock Market AI  Real-Time Forecasting System

Système de prédiction boursière en temps réel utilisant des architectures deep learning SOTA (LSTM, Transformers, TCN), conçu pour des inférences low-latency et une intégration transparente avec les flux de marché.

## Features

* **Prédictions multi-horizons** (1h, 1j, 7j)
* **Modèles embarquant l'attention** et mécanismes de mémoire à long terme
* Pipeline de données **temps réel** via WebSocket/Kafka
* Quantification dynamique des modèles pour **<50ms de latence**
* Intégration de **sentiment analysis** (FinBERT) et données macroéconomiques
* Monitoring des dérives conceptuelles (Concept Drift Detection)
