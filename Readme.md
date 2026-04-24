# World Model FPGA - Prototype Grid Routing

Ce projet est la première étape de conception d'un **World Model** (Ha & Schmidhuber, 2018) appliqué à l'automatisation du design de FPGA. L'objectif est d'entraîner un agent à comprendre la dynamique d'une grille de routage via une abstraction MiniGrid.

## Architecture V-M-C

Le système est une architecture **Vision–Mémoire–Contrôleur** distribuée en deux conteneurs Docker :

```
env_server ──REST API──► trainer
 (FastAPI +              (PyTorch)
  MiniGrid)
```

### Composants

| Modèle | Classe | Fichier | Rôle |
|---|---|---|---|
| V (Vision) | `VAE` | `models/vae.py` | Encodeur CNN — compresse 3×7×7 → z (dim 32) |
| M (Mémoire) | `MDNRNN` | `models/mdn_rnn.py` | LSTMCell + MDN — prédit p(z_{t+1} \| z_t, a_t) |
| C (Contrôleur) | `Controller` | `models/controller.py` | Linéaire — [z, h] → 7 actions discrètes |

### API REST (`env_server`, port 8000)

| Méthode | Endpoint | Corps | Réponse |
|---|---|---|---|
| `GET` | `/reset` | — | `{ "obs": [[[int]]] }` (7×7×3) |
| `POST` | `/step` | `{ "action": int }` | `{ "obs", "reward": float, "done": bool }` |

## Structure des fichiers

```
src/
├── models/
│   ├── __init__.py
│   ├── vae.py           # VAE (β-VAE, loss ELBO)
│   ├── mdn_rnn.py       # LSTM + Mixture Density Network
│   └── controller.py    # Contrôleur linéaire
├── server.py            # Serveur FastAPI + MiniGrid
├── utils.py             # Normalisation obs, retry HTTP, constantes
├── 1_collect_data.py    # Collecte 10 000 transitions aléatoires → data/
├── 2_train_world.py     # Entraîne VAE puis MDN-RNN → checkpoints/
└── 3_train_controller.py # REINFORCE sur le Contrôleur → checkpoints/
data/                    # Créé à l'exécution (monté en volume Docker)
checkpoints/             # Créé à l'exécution (monté en volume Docker)
```

## Lancer le projet

### Démarrage de l'infrastructure

```bash
# Construit les images et démarre env_server + trainer (en arrière-plan)
docker compose up -d --build
```

### Étape 1 — Collecte de données (~10 000 transitions)

```bash
docker compose exec trainer python src/1_collect_data.py
```

Les épisodes sont sauvegardés dans `data/episode_XXXXXX.npz` sur l'hôte (volume monté).

### Étape 2 — Entraînement du World Model (VAE + MDN-RNN)

```bash
# Pas besoin de env_server — travaille uniquement sur les données collectées
docker compose run --rm trainer python src/2_train_world.py
```

Produit `checkpoints/vae.pt` et `checkpoints/mdn_rnn.pt`.

### Étape 3 — Entraînement du Contrôleur (REINFORCE)

```bash
docker compose exec trainer python src/3_train_controller.py
```

Produit `checkpoints/controller.pt`, mis à jour tous les 100 épisodes.

### Sans Docker

```bash
# Terminal 1
python src/server.py

# Terminal 2 — modifier SERVER_URL = "http://localhost:8000" dans utils.py ou les scripts
python src/1_collect_data.py
python src/2_train_world.py
python src/3_train_controller.py
```

## Progression de l'implémentation

- [x] Serveur MiniGrid (FastAPI)
- [x] Modèle V — VAE (encodeur CNN, loss ELBO)
- [x] Modèle M — MDN-RNN (LSTM + Mixture of Gaussians)
- [x] Modèle C — Contrôleur linéaire (REINFORCE)
- [x] Pipeline de collecte de données
- [x] Pipeline d'entraînement modulaire (3 scripts)
- [ ] Calcul de la loss RL complète (reward model)
- [ ] Entraînement dans le monde imaginé (sans env_server)
- [ ] Évaluation quantitative (reward cumulatif, courbes)
