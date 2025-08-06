# 🏗️ Design Patterns - Stock-AI Project

Ce dossier contient les diagrammes PlantUML illustrant les **3 patrons de conception les plus utilisés** dans le projet Stock-AI.

## 📊 **Top 3 Design Patterns**

### **1. 🎯 Strategy Pattern**
**Fichier :** `Strategy_Pattern_Feature_Selection.puml`

**Description :** Le patron Strategy permet de sélectionner dynamiquement l'algorithme de sélection de features selon le type de modèle ML.

**Utilisation dans le projet :**
- **Feature Selection** : Différentes stratégies selon le modèle (LSTM, Prophet, XGBoost)
- **Input Formatting** : Formatage des données selon le modèle
- **Confidence Calculation** : Calcul de confiance selon le type de modèle

**Avantages :**
- ✅ Extensibilité : Ajout facile de nouveaux modèles
- ✅ Maintenabilité : Code modulaire et testable
- ✅ Réutilisabilité : Stratégies interchangeables

### **2. 🏭 Factory Pattern**
**Fichier :** `Factory_Pattern_Model_Registry.puml`

**Description :** Le patron Factory centralise la création d'objets (modèles ML, scalers) avec un registre dynamique.

**Utilisation dans le projet :**
- **Model Registry** : Création centralisée des modèles ML
- **Scaler Factory** : Création des scalers selon le type de modèle

**Avantages :**
- ✅ Encapsulation : Logique de création centralisée
- ✅ Flexibilité : Configuration dynamique
- ✅ Testabilité : Mock facile des objets

### **3. 📋 Template Method Pattern**
**Fichier :** `Template_Method_Pattern_Data_Processing.puml`

**Description :** Le patron Template Method définit le squelette des algorithmes tout en laissant les sous-classes implémenter les détails.

**Utilisation dans le projet :**
- **BaseDataProcessor** : Pipeline de traitement de données
- **BaseModel** : Processus d'entraînement des modèles

**Avantages :**
- ✅ Cohérence : Structure uniforme des pipelines
- ✅ Réutilisabilité : Code commun partagé
- ✅ Maintenabilité : Modifications centralisées

## 🏗️ **Architecture et Intégration**

### **Architecture Globale**
**Fichier :** `Architecture_Integration_Top_3_Patterns.puml`

**Description :** Vue d'ensemble de l'intégration des 3 patrons dans l'architecture microservices.

### **Flux de Données**
**Fichier :** `Data_Flow_With_Top_3_Patterns.puml`

**Description :** Séquence d'interactions montrant comment les 3 patrons sont utilisés dans le flux de données.

## 🎯 **Pertinence des Patrons**

### **Pourquoi ces 3 patrons sont les plus pertinents ?**

1. **Résolvent les vrais problèmes** du domaine ML
2. **Facilitent l'évolution** du système
3. **Améliorent la maintenabilité** du code
4. **Supportent l'architecture microservices**
5. **Permettent l'extensibilité** future

### **Domaine ML/IA**
- **Strategy Pattern** - Pour les différents algorithmes ML
- **Factory Pattern** - Pour créer les modèles et scalers
- **Template Method** - Pour les pipelines de traitement

### **Domaine Architecture**
- **Database per Service** - Isolation des données
- **Microservices Pattern** - Séparation des responsabilités
- **Observer Pattern** - Monitoring et métriques

## 📁 **Structure des Fichiers**

```
docs/design_patterns/
├── README.md                                    # Documentation
├── Strategy_Pattern_Feature_Selection.puml      # Strategy Pattern
├── Factory_Pattern_Model_Registry.puml          # Factory Pattern
├── Template_Method_Pattern_Data_Processing.puml # Template Method
├── Architecture_Integration_Top_3_Patterns.puml # Architecture globale
└── Data_Flow_With_Top_3_Patterns.puml          # Flux de données
```

## 🚀 **Utilisation**

### **Génération des Diagrammes**

Pour générer les diagrammes PNG à partir des fichiers PlantUML :

```bash
# Installation de PlantUML
java -jar plantuml.jar *.puml

# Ou avec Docker
docker run -v $(pwd):/workspace plantuml/plantuml *.puml
```

### **Visualisation**

Les diagrammes peuvent être visualisés :
- **Directement** dans les éditeurs supportant PlantUML
- **En ligne** sur [PlantUML Web Server](http://www.plantuml.com/plantuml/)
- **En local** avec l'extension PlantUML de votre IDE

## 📈 **Métriques d'Utilisation**

| Patron | Fichiers Utilisés | Services Impactés | Avantages Principaux |
|--------|-------------------|-------------------|---------------------|
| **Strategy** | 15+ fichiers | 6 services | Extensibilité ML |
| **Factory** | 8+ fichiers | 4 services | Création centralisée |
| **Template Method** | 12+ fichiers | 5 services | Structure commune |

## 🔄 **Évolution Future**

Ces patrons permettent d'ajouter facilement :
- **Nouveaux modèles ML** (Strategy)
- **Nouveaux types de scalers** (Factory)
- **Nouveaux pipelines de traitement** (Template Method)

---

**Version :** 1.0.0  
**Dernière mise à jour :** Janvier 2025  
**Auteur :** Équipe Stock-AI 
