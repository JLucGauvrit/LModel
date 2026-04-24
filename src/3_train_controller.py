"""
Étape 3 — Entraînement du Contrôleur par REINFORCE.

Le VAE et le MDN-RNN sont chargés depuis les checkpoints et gelés.
Seul le Contrôleur (une couche linéaire) est optimisé par policy gradient.

L'agent agit directement dans l'environnement réel via l'API env_server :
  1. V encode l'observation → z
  2. C choisit une action stochastique depuis Categorical(logits=C(z, h))
  3. L'action est envoyée à l'API, la récompense est collectée
  4. M met à jour l'état caché h (sans gradient, V et M gelés)
  5. Après l'épisode : G_t = Σ γ^k r_{t+k}, loss = -Σ log π(a_t) · G_t

Sortie :
  checkpoints/controller.pt

Usage :
  python src/3_train_controller.py
"""

import os

import requests
import torch

from models import VAE, MDNRNN, Controller
from utils import (
    obs_to_tensor, request_with_retry,
    LATENT_DIM, HIDDEN_DIM, ACTION_DIM, NUM_GAUSSIANS,
)

SERVER_URL = "http://env_server:8000"
CHECKPOINT_DIR = "checkpoints"

NUM_EPISODES = 1000
GAMMA = 0.99
LR = 1e-3
SAVE_EVERY = 100


def load_world_model(device: torch.device) -> tuple[VAE, MDNRNN]:
    """
    Charge et gèle le VAE et le MDN-RNN depuis les checkpoints.

    :param device: Device cible.
    :type device: torch.device
    :returns: Tuple (vae, mdn_rnn) en mode ``eval()``, poids gelés.
    :rtype: tuple[VAE, MDNRNN]
    :raises FileNotFoundError: Si un checkpoint est manquant.
    """
    paths = {
        "vae": os.path.join(CHECKPOINT_DIR, "vae.pt"),
        "mdn_rnn": os.path.join(CHECKPOINT_DIR, "mdn_rnn.pt"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint '{name}' introuvable : {path}\n"
                "Lancez d'abord 2_train_world.py."
            )

    vae = VAE(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(paths["vae"], map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    mdn_rnn = MDNRNN(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_gaussians=NUM_GAUSSIANS,
    ).to(device)
    mdn_rnn.load_state_dict(torch.load(paths["mdn_rnn"], map_location=device))
    mdn_rnn.eval()
    for p in mdn_rnn.parameters():
        p.requires_grad_(False)

    return vae, mdn_rnn


def compute_returns(rewards: list[float], gamma: float = GAMMA) -> torch.Tensor:
    """
    Calcule les retours actualisés G_t et les normalise pour réduire la variance.

    G_t = Σ_{k=0}^{T-t-1} γ^k · r_{t+k}

    :param rewards: Récompenses de l'épisode [r_0, r_1, ..., r_{T-1}].
    :type rewards: list[float]
    :param gamma: Facteur de dépréciation temporelle ∈ (0, 1].
    :type gamma: float
    :returns: Retours normalisés, shape (T,).
    :rtype: torch.Tensor
    """
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    ret = torch.tensor(returns, dtype=torch.float32)
    return (ret - ret.mean()) / (ret.std() + 1e-8)


def run_episode(
    session: requests.Session,
    vae: VAE,
    mdn_rnn: MDNRNN,
    controller: Controller,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    """
    Joue un épisode complet dans l'environnement et collecte les log-probs et récompenses.

    :param session: Session HTTP réutilisable.
    :type session: requests.Session
    :param vae: VAE gelé pour l'encodage des observations.
    :type vae: VAE
    :param mdn_rnn: MDN-RNN gelé pour la mise à jour de l'état caché.
    :type mdn_rnn: MDNRNN
    :param controller: Contrôleur en cours d'entraînement.
    :type controller: Controller
    :param device: Device de calcul.
    :type device: torch.device
    :returns: Tuple (log_probs stacked, rewards list).
    :rtype: tuple[torch.Tensor, list[float]]
    """
    res = request_with_retry(session, "GET", f"{SERVER_URL}/reset")
    obs = obs_to_tensor(res.json()["obs"], device)

    h, c = mdn_rnn.init_hidden(1, device)
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []

    done = False
    while not done:
        with torch.no_grad():
            z, _ = vae.encode(obs)

        action_logits = controller(z, h)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        res = request_with_retry(session, "POST", f"{SERVER_URL}/step",
                                 json={"action": action.item()})
        data = res.json()

        rewards.append(float(data["reward"]))
        done = data["done"]

        if not done:
            obs = obs_to_tensor(data["obs"], device)
            with torch.no_grad():
                _, _, _, h, c = mdn_rnn(z, action, h, c)

    return torch.stack(log_probs), rewards


def train_controller() -> None:
    """
    Boucle principale d'entraînement REINFORCE sur NUM_EPISODES épisodes.

    Affiche la récompense totale et la loss à chaque épisode.
    Sauvegarde le Contrôleur tous les SAVE_EVERY épisodes et en fin d'entraînement.

    :returns: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    vae, mdn_rnn = load_world_model(device)
    controller = Controller(
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_actions=ACTION_DIM
    ).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=LR)

    session = requests.Session()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"\nDébut entraînement Contrôleur — {NUM_EPISODES} épisodes...\n")

    for episode in range(1, NUM_EPISODES + 1):
        log_probs, rewards = run_episode(session, vae, mdn_rnn, controller, device)
        returns = compute_returns(rewards).to(device)

        loss = -(log_probs * returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Épisode {episode:4d}/{NUM_EPISODES} | Steps : {len(rewards):3d} | "
              f"Reward : {sum(rewards):6.3f} | Loss : {loss.item():7.4f}")

        if episode % SAVE_EVERY == 0:
            ckpt = os.path.join(CHECKPOINT_DIR, "controller.pt")
            torch.save(controller.state_dict(), ckpt)
            print(f"  → Checkpoint sauvegardé : {ckpt}\n")

    torch.save(controller.state_dict(), os.path.join(CHECKPOINT_DIR, "controller.pt"))
    print("\nEntraînement du Contrôleur terminé.")


if __name__ == "__main__":
    train_controller()
