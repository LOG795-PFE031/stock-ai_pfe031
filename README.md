# Guide d'utilisateur pour Stock-AI: Prédiction des prix d’actions en bourse

Bienvenue dans le guide d'utilisateur de notre projet! Ici, vous trouverez une documentation complète et détaillée sur différents aspects de notre projet.

Tout d'abord, Stock-AI est une plateforme complète de prédiction du cours des actions et d'analyse des sentiments utilisant des modèles d'apprentissage en profondeur (TensorFlow et PyTorch LSTM) et une interface de chatbot alimentée par l'IA.

## Table des matières

1. [Aperçu du projet](#aperçu)
2. [Comment utiliser ce guide](#comment-utiliser-ce-guide-dutilisateur)
3. [Architecture du système](#architecture-du-système)
4. [Installation](#installation)
5. [Rouler le système](#rouler-le-système)
6. [Tester les services](#tester-les-services)
7. [Troubleshooting](#troubleshooting)
8. [Fonctionnalités](#fonctionnalités)
9. [Remerciements](#remerciements)

## Comment utiliser ce guide d'utilisateur

- 🔗 **Liens utiles :** Consultez les liens surlignés en bleu vers d'autres parties du projet, des documents externes ou des ressources pertinentes.
- ⇽ **Retour vers le guide :** Pour retourner à ce guide.

## Aperçu 

Le système Stock-AI est une plateforme intégrée qui :

- Prédit le cours des actions grâce à des modèles d'apprentissage profond.
- Analyse l'opinion publique pour fournir des informations d'investissement.
- Traite les données via des files d'attente de messages distribuées.
- Fournit une interface de chatbot interactive pour les requêtes des utilisateurs.
- Propose une architecture de microservices pour une évolutivité et une résilience optimales.

Le système combine plusieurs technologies, dont PyTorch, TensorFlow, Docker et autres, pour créer une plateforme complète d'analyse boursière.

## Architecture du système

Le système est organisé autour d'une architecture de microservices conteneurisés avec Docker. Chaque composant est dédié à une fonction précise dans le pipeline d'analyse et de prédiction des marchés boursiers.

Les services principaux sont :

1. **🧠 Service d'ingestion de données (`data-service`)**  
   Récupère et stocke les données historiques des actions depuis des sources externes dans une base PostgreSQL dédiée.

2. **📰 Service d'analyse d'actualités (`news-service`)**  
   Collecte les actualités financières et effectue une analyse de sentiment via FinBERT.

3. **🧮 Service de prétraitement (`data-processing-service`)**  
   Nettoie, normalise et transforme les données brutes pour les rendre compatibles avec les modèles d'entraînement et d'inférence.

4. **🏋️‍♂️ Service d'entraînement de modèles (`training-service`)**  
   Entraîne des modèles LSTM, Prophet ou XGBoost à partir des données préparées et les enregistre dans MLflow.

5. **🚀 Service de prédiction (`deployment-service`)**  
   Charge les modèles en production et génère des prédictions futures sur les cours boursiers. Fournit également les scores de confiance.

6. **📈 Service d’évaluation (`evaluation-service`)**  
   Compare les prédictions aux valeurs réelles et calcule des métriques comme MAE, RMSE et R².

7. **📉 Service de monitoring (`monitoring-service`)**  
   Surveille les performances des modèles (drift, erreur, etc.) et déclenche automatiquement un réentraînement si nécessaire.

8. **🔀 Service d’orchestration (`orchestration-service`)**  
   Coordonne l’exécution des workflows de bout en bout (prétraitement → entraînement → prédiction → évaluation).

9. **🌐 API Gateway (`api-gateway`)**  
   Point d’entrée unique pour les utilisateurs. Expose toutes les fonctionnalités via une interface REST (FastAPI).

10. **📊 Services de suivi (`mlflow-server`, `mlflow-postgres`, `mlflow-minio`)**  
    Gèrent le suivi des expériences ML, le stockage des artefacts, et les métadonnées de modèles.

11. **⚙️ Services d’automatisation (`prefect-server`, `prefect-postgres`)**  
    Utilisés pour exécuter et planifier les workflows à l’aide de Prefect.

Tous les composants sont conteneurisés avec Docker pour un déploiement et une mise à l'échelle simple.

## Installation

### Prérequis

- Docker et Docker Compose
- Python 3.11+ (pour le développement local)
- Git

### Étape 1 : Configurer le backend
   1. Clonez le dépôt backend (service C#) à l'adresse https://github.com/LOG795-PFE031/BackendMicroservices_pfe031
   2. Suivez les étapes du fichier README de ce dépôt pour le mettre en service.

### Étape 2 : Installer le projet

1. **Cloner le dépôt**
   ```bash
   git clone <url-du-depot>
   ```

2. Assurez-vous d’être à la racine du projet, là où se trouve le fichier `docker-compose.yml`.

### Étape 3 : Configurer le Chatbot

Créez un fichier `.env` dans le dossier `/chatbot` avec votre clé API OpenAI :
```
OPENAI_API_KEY=your_api_key_here
```
 - *Vous devez avoir une carte de paiement reliée à votre compte.*

### Étape 4 : Installer le frontend 
   1. Clonez le dépôt frontend à l'adresse https://github.com/LOG795-PFE031/NotYahoo_pfe031
   2. Suivez les étapes du ficher README de ce dépôt pour le mettre en service.

## Rouler le système

### Créer et démarrer les services

Assurez-vous d’être à la racine du projet, là où se trouve le fichier `docker-compose.yml`.

Créez et exécutez tous les services avec Docker Compose :

```bash
docker compose up --build
```

**Note**: Initial build may take 10-30 minutes as PyTorch and other dependencies are installed.

⚠️ Si ce n’est pas votre premier build, supprimez d’abord le dossier `data/` pour éviter les conflits avec d’anciens artefacts : 
   ```bash
   rm -rf data
   docker-compose up --build
   ```

Patientez pendant le lancement, tous les services vont démarrer automatiquement. Vous pouvez ensuite accéder à l’API sur http://localhost:8000/docs.

### Démarrage de composants individuels

Si vous voulez démarrer des composants séparément :

**Service d'analyse d'actualités** :

```bash
docker compose up news-service
```
*Même logique pour les autres services.*

**Chatbot (localement)** :

```bash
cd chatbot
python chatbot.py
```

## Tester les services

### Service de prédiction boursière

Accéder à la documentation de l'interface utilisateur Swagger et à l'interface de test :

```
http://localhost:8000/docs
```

Exemples d'appels d'API :

```bash
curl -X GET "http://localhost:8000/api/predict/AAPL"
```

### Interface du chatbot

Testez le chatbot avec la commande curl :

```bash
curl -X POST http://localhost:5004/chat -H "Content-Type: application/json" -d '{"user_id": "test_user", "query": "Devrais-je investir dans MSFT ?"}'
```

Exemples de requêtes pour le chatbot :

- « Quelle est la prévision de cours pour AAPL ?»
- « Montrez-moi l'analyse de sentiment pour Tesla »
- « Devrais-je investir dans Google en ce moment ?»
- « Quel est le sentiment de l'actualité pour NVDA ? »

## Troubleshooting

### Problèmes de clé API du chatbot

Si le chatbot ne parvient pas à se connecter à OpenAI :

1. Vérifiez votre clé API dans le fichier « .env ».
2. Vérifiez les limites de débit OpenAI ou les modifications de l'API.

## Fonctionnalités

* **Prévision boursière multi-modèles** : Modèles LSTM TensorFlow, PyTorch, Prophet et Xgboost.
* **Chatbot optimisé par l'IA** : Interface en langage naturel utilisant OpenAI.
* **Conteneurisation Docker** : Déploiement et mise à l'échelle faciles.
* **Architecture de microservices** : Services indépendants et faiblement couplés.

## Remerciements

- OpenAI pour les fonctionnalités du chatbot
- yfinance et Stooq pour la fourniture de données boursières
- TensorFlow et PyTorch pour les frameworks d'apprentissage profond
- La communauté open source pour les différents outils et bibliothèques

---

## License

[MIT License](LICENSE)
