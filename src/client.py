import requests
import torch
import numpy as np
from model import VisionModel, MemoryModel, Controller

SERVER_URL = "http://env_server:8000"


def train() -> None:
    """
    Boucle principale d'entraînement de l'agent V-M-C sur 500 épisodes.

    À chaque épisode :

    1. L'environnement est réinitialisé via ``GET /reset``.
    2. Le modèle Vision (V) encode l'observation en vecteur latent.
    3. Le Contrôleur (C) choisit une action stochastique depuis la distribution Categorical.
    4. L'action est envoyée au serveur via ``POST /step``.
    5. Le modèle Mémoire (M) met à jour l'état caché LSTM avec l'observation et l'action.
    6. La boucle continue jusqu'à ce que ``done`` soit ``True``.

    .. note::
        Le calcul de la loss et l'étape d'optimisation ne sont pas encore implémentés.
        L'optimiseur Adam est initialisé mais ``optimizer.step()`` n'est pas appelé.

    :returns: None
    """
    latent_dim = 32
    hidden_dim = 64

    V = VisionModel(latent_dim=latent_dim)
    M = MemoryModel(latent_dim=latent_dim, hidden_dim=hidden_dim)
    C = Controller(latent_dim=latent_dim, hidden_dim=hidden_dim)

    optimizer = torch.optim.Adam(
        list(V.parameters()) + list(M.parameters()) + list(C.parameters()),
        lr=1e-3
    )

    print("Début de l'entraînement...")

    for episode in range(500):
        res = requests.get(f"{SERVER_URL}/reset").json()

        # MiniGrid renvoie (7,7,3) — PyTorch attend (Batch, Channels, H, W)
        obs_np = np.array(res['obs']).transpose(2, 0, 1)
        obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

        h_x = torch.zeros(1, hidden_dim)
        c_x = torch.zeros(1, hidden_dim)

        done = False
        step = 0
        total_reward = 0

        while not done:
            z = V(obs)

            action_logits = C(z, h_x)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()

            action_payload = {"action": action.item()}
            res = requests.post(f"{SERVER_URL}/step", json=action_payload).json()

            reward = res['reward']
            done = res['done']
            total_reward += reward

            next_obs_np = np.array(res['obs']).transpose(2, 0, 1)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).unsqueeze(0)

            h_x, c_x = M(z, action, h_x, c_x)

            obs = next_obs
            step += 1

        print(f"Épisode {episode} terminé en {step} étapes | Récompense totale : {total_reward}")


if __name__ == "__main__":
    train()
