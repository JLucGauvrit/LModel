# World Model FPGA - Prototype Grid Routing

Ce projet est la première étape de conception d'un **World Model** appliqué à l'automatisation du design de FPGA (Electronic Design Automation).

L'objectif actuel est d'entraîner un agent à comprendre la dynamique d'une grille de routage (via une abstraction MiniGrid) en séparant l'environnement de simulation du moteur d'entraînement.

## Architecture du Projet

Le système utilise une architecture **V-M-C** (Vision-Mémoire-Contrôleur) distribuée en deux conteneurs :

1. **`env_server` (Simulation)** : Serveur FastAPI hébergeant l'environnement MiniGrid. Il expose l'état du monde via une API REST.
2. **`trainer` (Intelligence)** : Client PyTorch gérant le modèle de dynamique (World Model) et l'optimisation des actions.

### Composants V-M-C (`src/model.py`)

| Modèle | Classe | Rôle |
|---|---|---|
| V (Vision) | `VisionModel` | Encodeur CNN — compresse l'image 3×7×7 en vecteur latent |
| M (Mémoire) | `MemoryModel` | LSTMCell — maintient l'état caché de la dynamique du monde |
| C (Contrôleur) | `Controller` | MLP — produit les logits d'action depuis (z, h) |

### API REST (`env_server`)

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/reset` | Réinitialise l'environnement, retourne `{ "obs": [[[int]]] }` |
| `POST` | `/step` | Exécute `{ "action": int }`, retourne `{ "obs", "reward", "done" }` |

## Structure des fichiers

```
.
├── src/
│   ├── server.py      # Environnement (FastAPI + MiniGrid)
│   ├── client.py      # Boucle d'entraînement (PyTorch)
│   └── model.py       # Définitions VisionModel, MemoryModel, Controller
├── docker-compose.yml # Orchestration des conteneurs
└── Dockerfile         # Image Python avec torch, gymnasium, minigrid
```

## Lancer le projet

```bash
docker-compose up
```

Cela démarre `env_server` (port 8000) puis `trainer` automatiquement.

**Sans Docker :**

```bash
# Terminal 1
python src/server.py

# Terminal 2 — modifier SERVER_URL dans client.py : "http://localhost:8000"
python src/client.py
```

## État actuel

- [x] Serveur MiniGrid fonctionnel avec API REST
- [x] Architecture V-M-C instanciée et forward pass complet
- [ ] Calcul de la loss (RL / VAE) — non encore implémenté
- [ ] Étape d'optimisation (`optimizer.step()`) — non encore implémentée
