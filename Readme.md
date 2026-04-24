# World Model FPGA - Prototype Grid Routing

Ce projet est la première étape de conception d'un **World Model** appliqué à l'automatisation du design de FPGA (Electronic Design Automation). 

L'objectif actuel est d'entraîner un agent à comprendre la dynamique d'une grille de routage (via une abstraction MiniGrid) en séparant l'environnement de simulation du moteur d'entraînement.

## 🏗️ Architecture du Projet

Le système utilise une architecture **V-M-C** (Vision-Mémoire-Contrôleur) distribuée en deux conteneurs :

1.  **`env_server` (Simulation)** : Serveur FastAPI hébergeant l'environnement MiniGrid. Il expose l'état du monde via une API REST.
2.  **`trainer` (Intelligence)** : Client PyTorch gérant le modèle de dynamique (World Model) et l'optimisation des actions.

## 📁 Structure des fichiers

```text
.
├── src/
│   ├── server.py      # Environnement (FastAPI + MiniGrid)
│   ├── client.py      # Boucle d'entraînement (PyTorch)
│   └── models/        # Définitions VAE, LSTM et Contrôleur
├── docker-compose.yml # Orchestration des conteneurs
└── Dockerfile         # Image Python optimisée
```
