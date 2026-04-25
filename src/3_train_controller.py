"""
Étape 3 — Entraînement du Contrôleur par REINFORCE.

Le VAE et le MDN-RNN sont chargés depuis les checkpoints et gelés.
Seul le Contrôleur (une couche linéaire) est optimisé par policy gradient.

L'agent agit directement dans l'environnement MiniGrid local :
  1. V encode l'observation → z
  2. C choisit une action stochastique depuis Categorical(logits=C(z, h))
  3. L'action est exécutée dans l'env local, la récompense est collectée
  4. M met à jour l'état caché h (sans gradient, V et M gelés)
  5. Après l'épisode : G_t = Σ γ^k r_{t+k}, loss = -Σ log π(a_t) · G_t

Sortie :
  checkpoints/controller.pt
  runs/controller/  ← logs TensorBoard

Usage :
  python src/3_train_controller.py
"""

import os
import time
from collections import deque

import gymnasium as gym
import minigrid  # noqa: F401 — enregistre les environnements MiniGrid
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models import VAE, MDNRNN, Controller
from utils import (
    obs_array_to_tensor,
    LATENT_DIM, HIDDEN_DIM, ACTION_DIM, NUM_GAUSSIANS,
    write_status,
)

CHECKPOINT_DIR = "checkpoints"
RUNS_DIR = "runs/controller"

NUM_EPISODES = 1000
GAMMA = 0.99
LR = 1e-3
SAVE_EVERY = 100
REWARD_WINDOW = 20  # taille de la fenêtre glissante pour la moyenne de récompense


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
        "vae":     os.path.join(CHECKPOINT_DIR, "vae.pt"),
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
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM, num_gaussians=NUM_GAUSSIANS,
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
    env: gym.Env,
    vae: VAE,
    mdn_rnn: MDNRNN,
    controller: Controller,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    """
    Joue un épisode complet dans l'environnement et collecte les log-probs et récompenses.

    :param env: Environnement MiniGrid local.
    :type env: gym.Env
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
    gym_obs, _ = env.reset()
    obs = obs_array_to_tensor(gym_obs["image"][np.newaxis], device)  # (1, 3, 7, 7)

    h, c = mdn_rnn.init_hidden(1, device)
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []

    done = False
    while not done:
        with torch.no_grad():
            z, _ = vae.encode(obs)

        action_logits = controller(z, h)
        dist   = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        gym_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        rewards.append(float(reward))

        if not done:
            obs = obs_array_to_tensor(gym_obs["image"][np.newaxis], device)
            with torch.no_grad():
                _, _, _, h, c = mdn_rnn(z, action, h, c)

    return torch.stack(log_probs), rewards


def train_controller() -> None:
    """
    Boucle principale d'entraînement REINFORCE sur NUM_EPISODES épisodes.

    Métriques loggées dans TensorBoard :
      - ``Controller/reward``       — récompense totale de l'épisode
      - ``Controller/reward_avg``   — moyenne glissante sur REWARD_WINDOW épisodes
      - ``Controller/episode_steps`` — nombre de pas de l'épisode
      - ``Controller/policy_loss``  — loss REINFORCE

    Sauvegarde le Contrôleur tous les SAVE_EVERY épisodes et en fin d'entraînement.

    :returns: None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device : {device} ({gpu_name})")

    vae, mdn_rnn = load_world_model(device)
    controller = Controller(
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_actions=ACTION_DIM
    ).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=LR)

    env    = gym.make("MiniGrid-Empty-8x8-v0")
    writer = SummaryWriter(log_dir=RUNS_DIR)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    reward_window: deque[float] = deque(maxlen=REWARD_WINDOW)
    print(f"TensorBoard → {RUNS_DIR}")
    print(f"\nDébut entraînement Contrôleur — {NUM_EPISODES} épisodes...\n")

    t_start = time.time()
    for episode in range(1, NUM_EPISODES + 1):
        log_probs, rewards = run_episode(env, vae, mdn_rnn, controller, device)
        returns = compute_returns(rewards).to(device)

        loss = -(log_probs * returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        reward_window.append(total_reward)
        avg_reward = sum(reward_window) / len(reward_window)

        writer.add_scalar("Controller/reward",        total_reward, episode)
        writer.add_scalar("Controller/reward_avg",    avg_reward,   episode)
        writer.add_scalar("Controller/episode_steps", len(rewards), episode)
        writer.add_scalar("Controller/policy_loss",   loss.item(),  episode)
        write_status(3, "Contrôleur — REINFORCE", episode, NUM_EPISODES,
                     {"reward_avg": round(avg_reward, 3)})

        elapsed   = time.time() - t_start
        avg_ep    = elapsed / episode
        remaining = avg_ep * (NUM_EPISODES - episode)
        eta       = f"{int(remaining // 3600):02d}:{int((remaining % 3600) // 60):02d}:{int(remaining % 60):02d}"
        print(f"[Ctrl] Ep {episode:4d}/{NUM_EPISODES} | "
              f"Steps {len(rewards):3d} | "
              f"Reward {total_reward:6.3f} | "
              f"Avg({REWARD_WINDOW}) {avg_reward:6.3f} | "
              f"Loss {loss.item():8.4f} | "
              f"{eta} restant")

        if episode % SAVE_EVERY == 0:
            ckpt = os.path.join(CHECKPOINT_DIR, "controller.pt")
            torch.save(controller.state_dict(), ckpt)
            print(f"  → Checkpoint : {ckpt}\n")

    torch.save(controller.state_dict(), os.path.join(CHECKPOINT_DIR, "controller.pt"))
    env.close()
    writer.close()
    print("\nEntraînement du Contrôleur terminé.")


if __name__ == "__main__":
    train_controller()
