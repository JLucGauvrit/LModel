"""
Étape 3 — Entraînement du Contrôleur par REINFORCE (dream rollouts).

Le VAE et le MDN-RNN sont chargés depuis les checkpoints et gelés.
Seul le Contrôleur (une couche linéaire) est optimisé par policy gradient.

Plutôt que d'interagir avec de vrais environnements (goulet CPU), le
Contrôleur s'entraîne entièrement dans l'espace latent du MDN-RNN :

  1. NUM_ENVS états initiaux z sont échantillonnés depuis les données collectées.
  2. À chaque pas « rêvé » :
       a. Controller(z, h) → distribution d'actions → action stochastique
       b. MDN-RNN(z, a, h) → z_{t+1} (échantillonné), r_t (prédit)
  3. Après DREAM_STEPS pas :
       G_t = Σ γ^k r_{t+k}
       loss = -Σ log π(a_t) · G_t  -  β · H(π)   (entropie régularisée)

Avantage : toutes les opérations restent sur GPU — pas d'attente Python/env.

Sortie :
  checkpoints/controller.pt
  runs/controller/  ← logs TensorBoard

Usage :
  python src/3_train_controller.py
"""

import glob
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter

from models import VAE, MDNRNN, Controller
from utils import (
    obs_array_to_tensor,
    LATENT_DIM, HIDDEN_DIM, ACTION_DIM, NUM_GAUSSIANS,
    write_status,
)

CHECKPOINT_DIR  = "checkpoints"
DATA_DIR        = "data"
RUNS_DIR        = "runs/controller"

NUM_ENVS        = 512   # rêves en parallèle par update
NUM_UPDATES     = 2000  # nombre de mises à jour du gradient
DREAM_STEPS     = 64    # pas de temps par rêve
GAMMA           = 0.99
LR              = 3e-4
ENTROPY_COEFF   = 0.01
MAX_GRAD_NORM   = 1.0
SAVE_EVERY      = 100
REWARD_WINDOW   = 20


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
    vae.load_state_dict(torch.load(paths["vae"], map_location=device, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    mdn_rnn = MDNRNN(
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM, num_gaussians=NUM_GAUSSIANS,
    ).to(device)
    ckpt = torch.load(paths["mdn_rnn"], map_location=device, weights_only=True)
    missing, _ = mdn_rnn.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"  ⚠ MDN-RNN : clés manquantes {missing} — reward_head non entraîné.")
    mdn_rnn.eval()
    for p in mdn_rnn.parameters():
        p.requires_grad_(False)

    return vae, mdn_rnn


def build_z_pool(vae: VAE, device: torch.device) -> torch.Tensor:
    """
    Construit un pool de vecteurs z initiaux depuis les premières observations
    de chaque épisode collecté.

    Ces z servent de points de départ pour les dream rollouts, garantissant
    que le Contrôleur explore un espace latent réaliste.

    :param vae: VAE gelé pour encoder les observations.
    :type vae: VAE
    :param device: Device cible.
    :type device: torch.device
    :returns: Pool de z, shape (N, latent_dim).
    :rtype: torch.Tensor
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*.npz")))
    if not files:
        raise FileNotFoundError(f"Aucune donnée dans {DATA_DIR}/. Lancez 1_collect_data.py.")
    z_list = []
    for f in files:
        data = np.load(f)
        obs_first = data["obs"][:1]   # première observation de chaque épisode
        x = obs_array_to_tensor(obs_first, device)
        with torch.no_grad():
            mu, _ = vae.encode(x)   # mode de la distribution (sans bruit)
        z_list.append(mu)
    pool = torch.cat(z_list, dim=0)   # (num_episodes, latent_dim)
    print(f"  Pool z : {pool.size(0)} points initiaux depuis {DATA_DIR}/")
    return pool


def compute_dream_returns(rewards: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    """
    Calcule les retours actualisés G_t sur un batch de rêves (vectorisé GPU).

    :param rewards: Récompenses, shape (T, N).
    :type rewards: torch.Tensor
    :param gamma: Facteur de dépréciation temporelle.
    :type gamma: float
    :returns: Retours normalisés par rêve, shape (T, N).
    :rtype: torch.Tensor
    """
    T, N = rewards.shape
    G = torch.zeros(N, device=rewards.device)
    returns = torch.zeros_like(rewards)
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    mean = returns.mean(dim=0, keepdim=True)
    std  = returns.std(dim=0, keepdim=True).clamp(min=1e-5)
    return (returns - mean) / (std + 1e-8)


def collect_dream_rollouts(
    mdn_rnn: MDNRNN,
    controller: Controller,
    device: torch.device,
    z_pool: torch.Tensor,
    num_envs: int,
    dream_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simule ``num_envs`` rêves de ``dream_steps`` pas dans l'espace latent.

    Aucun environnement réel n'est instancié : toutes les opérations se
    déroulent sur GPU. Toutes les dimensions batch sont traitées en parallèle,
    sans aucun loop Python par environnement.

    :param mdn_rnn: MDN-RNN gelé avec reward_head.
    :type mdn_rnn: MDNRNN
    :param controller: Contrôleur en cours d'entraînement.
    :type controller: Controller
    :param device: Device de calcul.
    :type device: torch.device
    :param z_pool: Pool de z initiaux, shape (P, latent_dim).
    :type z_pool: torch.Tensor
    :param num_envs: Nombre de rêves en parallèle.
    :type num_envs: int
    :param dream_steps: Nombre de pas par rêve.
    :type dream_steps: int
    :returns: Tenseurs (log_probs, entropies, rewards), chacun shape (T, N).
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    idx = torch.randint(len(z_pool), (num_envs,))
    z   = z_pool[idx].to(device)
    h, c = mdn_rnn.init_hidden(num_envs, device)

    lp_steps: list[torch.Tensor] = []
    en_steps: list[torch.Tensor] = []
    r_steps:  list[torch.Tensor] = []

    for _ in range(dream_steps):
        logits  = controller(z, h)
        dist    = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        lp_steps.append(dist.log_prob(actions))   # (N,)
        en_steps.append(dist.entropy())            # (N,)

        with torch.no_grad():
            pi, mu, sigma, r_pred, h, c = mdn_rnn(z, actions, h, c)
            z = MDNRNN.sample_latent(pi, mu, sigma)

        r_steps.append(r_pred.squeeze(-1))         # (N,)

    # (T, N) — aucun loop per-env
    return (
        torch.stack(lp_steps),
        torch.stack(en_steps),
        torch.stack(r_steps),
    )


def train_controller() -> None:
    """
    Boucle principale d'entraînement REINFORCE sur NUM_UPDATES mises à jour.

    Chaque update collecte NUM_ENVS rêves en parallèle dans l'espace latent,
    calcule la loss REINFORCE avec régularisation entropie, et effectue un pas
    de gradient.

    Métriques loggées dans TensorBoard :
      - ``Controller/reward_avg``    — moyenne glissante des récompenses prédites
      - ``Controller/episode_steps`` — dream_steps (constant, informatif)
      - ``Controller/policy_loss``   — loss REINFORCE

    :returns: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device : {device} ({gpu_name})")

    torch.backends.cudnn.benchmark = True

    vae, mdn_rnn = load_world_model(device)
    z_pool = build_z_pool(vae, device)

    controller = Controller(
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_actions=ACTION_DIM
    ).to(device)

    optimizer = torch.optim.Adam(controller.parameters(), lr=LR)

    writer = SummaryWriter(log_dir=RUNS_DIR)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    reward_window: deque[float] = deque(maxlen=REWARD_WINDOW)
    total_episodes = 0

    print(f"Rêves parallèles : {NUM_ENVS}   Pas/rêve : {DREAM_STEPS}   Mises à jour : {NUM_UPDATES}")
    print(f"TensorBoard → {RUNS_DIR}\n")

    t_start = time.time()
    for update in range(1, NUM_UPDATES + 1):

        # log_probs, entropies, rewards : shape (T, N) sur GPU
        log_probs, entropies, rewards = collect_dream_rollouts(
            mdn_rnn, controller, device, z_pool, NUM_ENVS, DREAM_STEPS
        )

        returns = compute_dream_returns(rewards)  # (T, N), normalisé par rêve

        # REINFORCE : loss moyennée sur T×N transitions
        policy_loss  = -(log_probs * returns).sum() / NUM_ENVS
        entropy_loss = -ENTROPY_COEFF * entropies.mean()
        loss = policy_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn_utils.clip_grad_norm_(controller.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_episodes += NUM_ENVS
        avg_ep_reward   = rewards.sum(dim=0).mean().item() / DREAM_STEPS
        reward_window.append(avg_ep_reward)
        avg_reward      = sum(reward_window) / len(reward_window)

        writer.add_scalar("Controller/reward_avg",    avg_reward,  total_episodes)
        writer.add_scalar("Controller/episode_steps", DREAM_STEPS, total_episodes)
        writer.add_scalar("Controller/policy_loss",   loss.item(), total_episodes)
        write_status(3, "Contrôleur — REINFORCE (dream)", update, NUM_UPDATES,
                     {"reward_avg": round(avg_reward, 3)})

        elapsed   = time.time() - t_start
        avg_up    = elapsed / update
        remaining = avg_up * (NUM_UPDATES - update)
        eta = (f"{int(remaining // 3600):02d}:"
               f"{int((remaining % 3600) // 60):02d}:"
               f"{int(remaining % 60):02d}")
        print(f"[Ctrl] Update {update:4d}/{NUM_UPDATES} | "
              f"Ep {total_episodes:6d} | "
              f"Reward {avg_ep_reward:6.4f} | "
              f"Avg({REWARD_WINDOW}) {avg_reward:6.4f} | "
              f"Loss {loss.item():8.4f} | "
              f"{eta} restant")

        if update % SAVE_EVERY == 0:
            ckpt = os.path.join(CHECKPOINT_DIR, "controller.pt")
            torch.save(controller.state_dict(), ckpt)
            print(f"  → Checkpoint : {ckpt}\n")

    torch.save(controller.state_dict(), os.path.join(CHECKPOINT_DIR, "controller.pt"))
    writer.close()
    print("\nEntraînement du Contrôleur terminé.")


if __name__ == "__main__":
    train_controller()
