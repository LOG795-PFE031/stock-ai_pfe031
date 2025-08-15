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
7. [Dépannage](#dépannage)
8. [Fonctionnalités](#fonctionnalités)
9. [Remerciements](#remerciements)

## Comment utiliser ce guide d'utilisateur

- 🔗 **Liens utiles :** Consultez les liens surlignés en bleu vers d'autres parties du projet, des documents externes ou des ressources pertinentes.
- ⇽ **Retour vers le guide :** Pour retourner à ce guide.
- ⇧ **Retour vers la table des matières :** Pour retourner à la table des matières rapidement.

## Aperçu 
[⇧ Retour à la table des matières](#table-des-matières)

Le système Stock-AI est une plateforme intégrée qui :

- Prédit le cours des actions grâce à des modèles d'apprentissage profond.
- Analyse l'opinion publique pour fournir des informations d'investissement.
- Traite les données via des files d'attente de messages distribuées.
- Fournit une interface de chatbot interactive pour les requêtes des utilisateurs.
- Propose une architecture de microservices pour une évolutivité et une résilience optimales.

Le système combine plusieurs technologies, dont PyTorch, TensorFlow, Docker et autres, pour créer une plateforme complète d'analyse boursière.

## Architecture du système
[⇧ Retour à la table des matières](#table-des-matières)

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
   Note : un job d’initialisation (`minio-create-bucket`) crée le bucket d’artefacts au démarrage.

12. **⚙️ Services d’orchestration de workflows (`prefect-server`, `prefect-postgres`)**  
   Utilisés pour exécuter et planifier les workflows à l’aide de Prefect.

13. **📡 Service d'observabilité avec Prometheus & Grafana (`prometheus`, `grafana`)**  
   Collecte de métriques d’infrastructure et d’application (Prometheus) et tableaux de bord (Grafana) pour la supervision en temps réel.

14. **🔐 Authentification & stockage associé (`auth-service`, `postgres`)**  
   Service d’authentification en C# et base Postgres associée pour les utilisateurs/portfolio.

15. **📨 Messagerie inter‑services (`rabbitmq`)**  
   Courtier de messages pour la communication asynchrone.

17. **💻 Frontend (`frontend`)**  
   Interface utilisateur (Vite/React) exposant les fonctionnalités de la plateforme.

Tous les composants sont conteneurisés avec Docker pour un déploiement et une mise à l'échelle simple.

## Installation
[⇧ Retour à la table des matières](#table-des-matières)

### Prérequis

- Docker et Docker Compose
- Python 3.11+ (pour le développement local)
- Git


### Étape 1 : Installer le projet

1. **Cloner le dépôt**
   ```bash
   git clone <url-du-depot>
   ```

2. Assurez-vous d’être à la racine du projet, là où se trouve le fichier `docker-compose.yml`.

### Étape 2 : Configurer le Chatbot

Créez un fichier `.env` dans le dossier `/chatbot` avec votre clé API OpenAI :
```
OPENAI_API_KEY=your_api_key_here
```
 - *Vous devez avoir une carte de paiement reliée à votre compte.*

### Étape 3 : Configurer le backend
   - Assurez-vous d'avoir mis à jour votre SDK DotNet vers .Net 8 (https://dotnet.microsoft.com/en-us/download/dotnet/8.0)
   - Assurez-vous d'avoir installé la dernière version de Visual Studio. Sinon, téléchargez l'installateur de Visual Studio et mettez-le à jour (assurez-vous qu'ASP.NET Core est installé).

   **Si vous avez effectué l'une des opérations ci-dessus, redémarrez votre ordinateur**

   1. À la racine de `/auth`, ouvrez un terminal (powershell de préférence)
   2. Roulez les commandes suivantes:
      ```
      dotnet dev-certs https --clean
      dotnet dev-certs https --trust
      dotnet dev-certs https -ep ./AuthNuget/AuthNuget/localhost.pfx -p "secret"
      ```
   3. Allez dans `Configuration/certs/` ou `\AuthNuget` manuellement et cliquez sur le fichier `localhost.pfx` et installez-le sur votre système.

### Étape 4 :Configuration du frontend 
1. **Configuration des variables d’environnement**
   Créez `frontend/.env`:
   ```bash
   VITE_PREDICTION_SERVICE_URL=http://localhost:8000
   VITE_SENTIMENT_SERVICE_URL=http://localhost:8092
   VITE_OPENAI_API_KEY=your_openai_api_key
   ```

   ***Développement local du frontend (sans Docker)***
   **Prérequis**:
   - [Bun](https://bun.sh/) (dernière version)
   ```
      cd frontend
      bun install
      bun run dev --host 0.0.0.0 --port 5173
   ```

## Rouler le système
[⇧ Retour à la table des matières](#table-des-matières)

### Créer et démarrer les services

Assurez-vous d’être à la racine du projet, là où se trouve le fichier `docker-compose.yml`.

Créez et exécutez tous les services avec Docker Compose :

```bash
docker-compose up --build
```

**Note**: La construction initiale peut prendre 10 à 30 minutes pendant que PyTorch et d'autres dépendances sont installées.

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
[⇧ Retour à la table des matières](#table-des-matières)

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

## Dépannage
[⇧ Retour à la table des matières](#table-des-matières)

### Stockage Docker plein

Il peut arriver souvent que l'espace de votre pc soit remplit par Docker, même si vous supprimez les images ou volumes de Docker. 
Suivez ces instructions pour réduire la taille de Docker sur votre disque.

##### 1. Avec Docker d'ouvert, entrer cette commande dans le terminal du projet :
```bash
docker system prune -a --volumes -f
```
Cela efface :
 * Tous les conteneurs arrêtés
 * Toutes les images inutilisées (pas seulement celles qui traînent)
 * Tous les réseaux inutilisés
 * Tout le cache de build
 * Tous les volumes inutilisés

*Exécutez-le avec prudence. Assurez-vous que vos conteneurs ou volumes ne stockent rien d'important avant de l'exécuter.*

##### 2. Arrêter le Docker
Ouvrez le gestionnaire de tâches pour vous assurer que tous les processus Docker sont fermés

##### 3. Ouvrir PowerShell en tant qu'administrateur
```bash
   wsl --shutdown
```
Cette commande arrête complètement le backend WSL2 et supprime les modifications du système de fichiers.

##### 4. Optimiser le disque virtuel WSL 
C'est ce qui réduit réellement le fichier `.vhdx` qui stocke les fichiers de Docker et qui prend autant de place.

*Vous devez faire ces étapes une à la fois.*

```bash
   diskpart
```

```bash
   # Remplacer `<YourUser>` par ton user :
   select vdisk file="C:\Users\<YourUser>\AppData\Local\Docker\wsl\data\ext4.vhdx"
```

```bash
   attach vdisk readonly
```

```bash
   compact vdisk
```

```bash
   detach vdisk
```

```bash
   exit
```

**Redémarrez Docker Desktop.
Votre disque devrait maintenant afficher l'espace disque récupéré. Sinon redémarrez votre ordinateur.**

### Problèmes de clé API du chatbot

Si le chatbot ne parvient pas à se connecter à OpenAI :

1. Vérifiez votre clé API dans le fichier `.env`.
2. Vérifiez les limites de débit OpenAI ou les modifications de l'API.

## Fonctionnalités
[⇧ Retour à la table des matières](#table-des-matières)

* **Prévision boursière multi-modèles** : Modèles LSTM TensorFlow, PyTorch, Prophet et Xgboost.
* **Chatbot optimisé par l'IA** : Interface en langage naturel utilisant OpenAI.
* **Conteneurisation Docker** : Déploiement et mise à l'échelle faciles.
* **Architecture de microservices** : Services indépendants et faiblement couplés.

## Remerciements
[⇧ Retour à la table des matières](#table-des-matières)

- OpenAI pour les fonctionnalités du chatbot
- yfinance et Stooq pour la fourniture de données boursières
- TensorFlow et PyTorch pour les frameworks d'apprentissage profond
- La communauté open source pour les différents outils et bibliothèques

---

## License

[MIT License](LICENSE)
