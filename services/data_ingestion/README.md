# Data Service Microservice

## Description

Le microservice Data Service est responsable de la récupération, du stockage et de la gestion des données financières brutes. Ce service est crucial pour fournir les données de base nécessaires à l'analyse et aux prédictions du marché boursier.

## Fonctionnalités

- **Récupération de données boursières** : Obtention des prix historiques et actuels des actions
- **Gestion des noms d'entreprises** : Mise en cache et récupération des noms d'entreprises
- **Stockage en base de données** : Persistance des données dans PostgreSQL
- **API REST** : Interface HTTP pour accéder aux données
- **Monitoring** : Métriques Prometheus et health checks
- **Gestion des erreurs** : Gestion robuste des erreurs et des timeouts

## Endpoints API

### Système
- `GET /data/health` - Vérification de l'état du service
- `GET /data/welcome` - Message de bienvenue et informations sur l'API
- `GET /data/metrics` - Métriques Prometheus

### Données
- `GET /data/stocks` - Liste des actions disponibles
- `GET /data/stock/current` - Récupération du prix actuel d'une action
- `GET /data/stock/historical` - Récupération des données boursières historiques
- `GET /data/stock/recent` - Récupération des données récentes (nombre de jours spécifié)
- `GET /data/stock/from-end-date` - Récupération des données depuis une date de fin
- `POST /data/cleanup` - Nettoyage des données (optionnel par symbole)

## Paramètres des endpoints

### GET /data/stocks
- Aucun paramètre requis

### GET /data/stock/current
- `symbol` (requis) : Symbole boursier (ex: AAPL, MSFT)

### GET /data/stock/historical
- `symbol` (requis) : Symbole boursier (ex: AAPL, MSFT)
- `start_date` (requis) : Date de début (YYYY-MM-DD)
- `end_date` (requis) : Date de fin (YYYY-MM-DD)

### GET /data/stock/recent
- `symbol` (requis) : Symbole boursier (ex: AAPL, MSFT)
- `days_back` (requis) : Nombre de jours en arrière depuis aujourd'hui (1-365)

### GET /data/stock/from-end-date
- `symbol` (requis) : Symbole boursier (ex: AAPL, MSFT)
- `end_date` (requis) : Date de fin (YYYY-MM-DD)
- `days_back` (requis) : Nombre de jours en arrière depuis la date de fin (1-365)

### POST /data/cleanup
- `symbol` (optionnel) : Symbole spécifique à nettoyer

## Architecture

Le service suit l'architecture des autres microservices du projet :

- **FastAPI** : Framework web moderne et rapide
- **SQLAlchemy** : ORM pour la gestion de la base de données
- **yfinance** : Bibliothèque pour récupérer les données Yahoo Finance
- **Prometheus** : Métriques et monitoring
- **Docker** : Conteneurisation

## Démarrage

### Avec Docker Compose
```bash
docker-compose up data-service
```

### En local
```bash
cd services/data_ingestion
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Tests

Pour tester le service :
```bash
python test_data_service.py
```

## Configuration

Le service utilise la configuration du projet principal via le module `core.config`. Les variables d'environnement importantes :

- `POSTGRES_HOST` : Hôte de la base de données PostgreSQL
- `POSTGRES_PORT` : Port de la base de données
- `POSTGRES_USER` : Utilisateur de la base de données
- `POSTGRES_PASSWORD` : Mot de passe de la base de données
- `POSTGRES_DB` : Nom de la base de données

## Monitoring

Le service expose des métriques Prometheus sur `/data/metrics` et un health check sur `/data/health`.

## Logs

Les logs sont gérés via le module `core.logging` avec le logger "data". 
