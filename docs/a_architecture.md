# 4+1 Architecture Development and Physical Views – Stock-ai Microservice

## 1. Development View
```plantuml
@startuml

package "StockAI" {

    package "API Gateway" {
        component "API Gateway" as API
    }

    package "Core Services" {
        component "News Service"
        component "Data Service"
        component "Prediction Service"
        component "Training Service"
    }
    
    package "Model" {
        component "Model Service"
        component "Trainer lstm"
        component "Trainer prophet"
    }
    
    package "Communication" {
        component "RabbitMq Service"
    }
    
    package "Monitoring" {
        component "Prometheus"
        component "Grafana"
    }
    
    "API" --> "Core Services"
    
    "Training Service" --> "Model"
    
    "Prediction Service" --> "Training Service"
    
    "Prediction Service" --> "RabbitMq Service"
}
@enduml
```

## 2. Physical View
```plantuml
@startuml
allowmixing

node "Stock-ai Cluster" {

  node "API Gateway" {
    component "API Gateway" as API
  }

  queue "Orchestrator"
  
  node "Core services"
  
  node "Monitoring"
  queue "RabbitMQ"
  database "Redis Cache" 
}

node "Backend_Microservice"

API --> "Orchestrator"
"Orchestrator" --> "Core services"
"Core services" --> "RabbitMQ"
"RabbitMQ" --> "Backend_Microservice"
"Monitoring" --> "Core services"
' "Backend_Microservice" --> "API Gateway"
@enduml
