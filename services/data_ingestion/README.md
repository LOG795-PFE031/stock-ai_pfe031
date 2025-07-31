# Data Service Microservice

## Description

Le microservice Data Service est responsable de la récupération, du stockage et de la gestion des données financières brutes. Ce service est crucial pour fournir les données de base nécessaires à l'analyse et aux prédictions du marché boursier. Il fait partie de l'architecture microservices du projet Stock-AI et communique avec les autres services via des API REST.

## Fonctionnalités

- **Récupération de données boursières** : Obtention des prix historiques et actuels des actions via Yahoo Finance
- **Gestion des noms d'entreprises** : Mise en cache et récupération des noms d'entreprises depuis stock_names.csv
- **Base de données dédiée** : Stockage dans une base PostgreSQL dédiée pour les données boursières
- **API REST FastAPI** : Interface HTTP moderne avec documentation automatique
- **Monitoring avancé** : Métriques Prometheus complètes (CPU, mémoire, requêtes, erreurs)
- **Gestion des erreurs** : Gestion robuste avec retry logic et timeouts configurables
- **Health checks** : Vérifications de santé multi-niveaux (service, base de données)
- **Containerisation** : Service entièrement containerisé avec Docker

## Architecture Technique

- **Framework** : FastAPI avec validation Pydantic
- **Base de données** : PostgreSQL dédiée avec SQLAlchemy async
- **Port** : 8001 (externe) → 8000 (interne)
- **Monitoring** : Prometheus + métriques custom
- **Logs** : Logging structuré via le module core
- **Tests** : Tests unitaires et d'intégration complets

## Endpoints API

### Système
- `GET /health` - Health check global du service
- `GET /data/health` - Vérification détaillée de l'état du service et composants
- `GET /data/welcome` - Message de bienvenue et informations sur l'API
- `GET /metrics` - Métriques Prometheus pour monitoring

### Données
- `GET /data/stocks` - Liste des actions disponibles avec informations détaillées
- `GET /data/stock/current` - Prix actuel d'une action avec métadonnées
- `GET /data/stock/historical` - Données historiques entre deux dates
- `GET /data/stock/recent` - Données récentes (X jours depuis aujourd'hui)
- `GET /data/stock/from-end-date` - Données depuis une date de fin spécifiée
- `POST /data/cleanup` - Nettoyage des données (global ou par symbole)

## Paramètres des endpoints

### GET /data/stocks
- Aucun paramètre requis
- **Retourne** : Liste des stocks avec symbole, nom, prix actuel et variation

### GET /data/stock/current
- `symbol` (requis) : Symbole boursier (ex: AAPL, MSFT)
- **Retourne** : Prix actuel, pourcentage de variation, timestamp, métadonnées

### GET /data/stock/historical
- `symbol` (requis) : Symbole boursier (ex: AAPL, MSFT)
- `start_date` (requis) : Date de début (YYYY-MM-DD)
- `end_date` (requis) : Date de fin (YYYY-MM-DD)
- **Retourne** : Liste des prix historiques avec métadonnées

### GET /data/stock/recent
- `symbol` (requis) : Symbole boursier (ex: AAPL, MSFT)
- `days_back` (requis) : Nombre de jours en arrière depuis aujourd'hui (1-365)
- **Retourne** : Données récentes avec informations de stock

### GET /data/stock/from-end-date
- `symbol` (requis) : Symbole boursier (ex: AAPL, MSFT)
- `end_date` (requis) : Date de fin (YYYY-MM-DD)
- `days_back` (requis) : Nombre de jours en arrière depuis la date de fin (1-365)
- **Retourne** : Données historiques depuis une date spécifique

### POST /data/cleanup
- `symbol` (optionnel) : Symbole spécifique à nettoyer
- **Retourne** : Statistiques de nettoyage (records supprimés, symboles affectés)

## Modèles de données (Schemas Pydantic)

### StockDataResponse
```json
{
  "symbol": "AAPL",
  "stock_info": {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "current_price": 150.25,
    "change_percent": 2.5
  },
  "prices": [
    {
      "date": "2024-01-15",
      "open": 148.50,
      "high": 152.00,
      "low": 147.25,
      "close": 150.25,
      "volume": 58234567,
      "adj_close": 150.25
    }
  ],
  "total_records": 252,
  "meta": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "version": "1.0.0"
  }
}
```

### HealthResponse
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "database": true,
    "data_service": true,
    "external_apis": true
  }
}
```

## Configuration Base de Données

Le service utilise une base PostgreSQL dédiée avec les paramètres suivants :

### Variables d'environnement
- `STOCK_DB_HOST` : postgres-stock-data (défaut)
- `STOCK_DB_PORT` : 5432
- `STOCK_DB_USER` : stockuser
- `STOCK_DB_PASSWORD` : stockpass
- `STOCK_DB_NAME` : stockdata
- `API_HOST` : 0.0.0.0
- `API_PORT` : 8000

### Tables de la base
- **stock_prices** : Stockage des prix historiques
  - id, symbol, date, open, high, low, close, volume, adj_close
  - Index optimisés pour les requêtes par symbole et date

## Monitoring et Métriques

### Métriques Prometheus disponibles
- `http_requests_total` : Nombre total de requêtes HTTP
- `http_request_duration_seconds` : Durée des requêtes HTTP
- `http_errors_total` : Nombre d'erreurs HTTP
- `cpu_usage_percent` : Utilisation CPU en temps réel
- `memory_usage_bytes` : Utilisation mémoire
- `external_requests_total` : Requêtes vers APIs externes (Yahoo Finance)

### Logs structurés
- Format JSON avec niveaux (DEBUG, INFO, WARNING, ERROR)
- Corrélation des requêtes avec trace IDs
- Logs d'accès et d'erreurs séparés

## Démarrage et Développement

### Avec Docker Compose (Recommandé)
```bash
# Démarrer uniquement le service data
docker-compose up data-service

# Démarrer avec les dépendances (base de données)
docker-compose up postgres-stock-data data-service

# Build et démarrage
docker-compose up --build data-service
```

### En local (Développement)
```bash
cd services/data_ingestion

# Installer les dépendances
pip install -r requirements.txt
pip install -r ../../core/requirements.txt

# Configurer les variables d'environnement
export PYTHONPATH="../../:$PYTHONPATH"
export STOCK_DB_HOST=localhost
export STOCK_DB_PORT=5432
# ... autres variables

# Démarrer le service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### URLs d'accès
- **API locale** : http://localhost:8001
- **Documentation Swagger** : http://localhost:8001/docs  
- **ReDoc** : http://localhost:8001/redoc
- **Métriques Prometheus** : http://localhost:8001/metrics
- **Health Check** : http://localhost:8001/health

## Tests et Validation

### Tests automatisés
```bash
# Tests unitaires
cd services/data_ingestion
python -m pytest tests/test_unitaire_data_service.py -v

# Tests d'intégration (nécessite containers)
python -m pytest tests/test_integration_data_service.py -v

# Test microservice complet
python tests/test_data_microservice.py
```

### Tests manuels avec curl
```bash
# Health check
curl http://localhost:8001/health

# Liste des stocks
curl http://localhost:8001/data/stocks

# Prix actuel d'Apple
curl "http://localhost:8001/data/stock/current?symbol=AAPL"

# Données historiques
curl "http://localhost:8001/data/stock/historical?symbol=AAPL&start_date=2024-01-01&end_date=2024-01-31"
```

## Architecture et Communication

### Communication inter-services
Le service expose des endpoints REST pour communication avec :
- **API Gateway** : Routage des requêtes clients
- **Training Service** : Récupération de données pour entraînement
- **News Service** : Synchronisation des données de marché
- **Monitoring** : Collecte de métriques

### Structure des fichiers
```
services/data_ingestion/
├── main.py                 # Point d'entrée FastAPI
├── routes.py              # Routes et endpoints API
├── data_service.py        # Logique métier principale
├── schemas.py             # Modèles Pydantic
├── requirements.txt       # Dépendances Python
├── Dockerfile            # Configuration container
├── docker-entrypoint.sh  # Script de démarrage
├── db/                   # Couche base de données
│   ├── session.py        # Sessions async SQLAlchemy
│   ├── init_db.py       # Initialisation DB
│   └── models/          # Modèles SQLAlchemy
└── tests/               # Tests complets
    ├── test_unitaire_data_service.py
    ├── test_integration_data_service.py
    └── test_data_microservice.py
```

## Troubleshooting et FAQ

### Problèmes courants

#### Service ne démarre pas
1. **Vérifier les variables d'environnement** :
   ```bash
   docker-compose logs data-service
   ```

2. **Base de données non accessible** :
   ```bash
   # Vérifier que postgres-stock-data est démarré
   docker-compose ps postgres-stock-data
   ```

#### Erreurs de connexion à Yahoo Finance
- **Timeout** : Le service implémente un retry automatique
- **Limite de taux** : Attendre 1-2 minutes entre les requêtes massives
- **Symbole invalide** : Vérifier dans `/data/stocks`

#### Performance lente
- **Indices manquants** : Recréer la base avec `init_db.py`
- **Cache des noms** : Vérifier `data/stock_names.csv`
- **Mémoire insuffisante** : Augmenter la limite Docker

### Logs et Debug
```bash
# Logs du service
docker-compose logs -f data-service

# Logs avec timestamps
docker-compose logs -f -t data-service

# Debug d'une requête spécifique
curl -v "http://localhost:8001/data/stock/current?symbol=AAPL"
```

## Sécurité et Bonnes Pratiques

### Sécurité
- **Variables sensibles** : Utiliser des secrets Docker/Kubernetes
- **CORS** : Configuré pour tous les origins (à restreindre en prod)
- **Rate limiting** : À implémenter pour les APIs publiques
- **SSL/TLS** : À configurer au niveau du reverse proxy

### Performance
- **Cache** : Les noms de stocks sont mis en cache
- **Pagination** : Implémentée sur les endpoints retournant beaucoup de données  
- **Connexions DB** : Pool de connexions async optimisé
- **Monitoring** : Métriques complètes pour identifier les goulots

### Maintenance
- **Backup DB** : Sauvegardes régulières de postgres-stock-data
- **Nettoyage** : Endpoint `/data/cleanup` pour purger les vieilles données
- **Mise à jour** : Rolling updates via Docker Compose
- **Monitoring** : Alertes sur les métriques critiques

---

## Support et Contact

Pour toute question ou problème :
- **Documentation technique** : `/docs` (Swagger UI)
- **Métriques** : `/metrics` (Prometheus)
- **Health checks** : `/health`
- **Tests** : Dossier `tests/` pour exemples d'utilisation

**Version** : 1.0.0  
**Dernière mise à jour** : Janvier 2025 
