"""
Étape 2 — Entraînement du World Model (VAE puis MDN-RNN).

Phase 1 : entraîne le VAE sur toutes les observations collectées.
Phase 2 : encode les séquences avec le VAE gelé, entraîne le MDN-RNN
          à prédire la distribution du prochain état latent z_{t+1}.

Sorties :
  checkpoints/vae.pt
  checkpoints/mdn_rnn.pt
  runs/world_model/  ← logs TensorBoard

Usage :
  python src/2_train_world.py
"""

import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models import VAE, MDNRNN
from utils import LATENT_DIM, HIDDEN_DIM, ACTION_DIM, NUM_GAUSSIANS, obs_array_to_tensor, write_status

DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
RUNS_DIR = "runs/world_model"

# --- Hyperparamètres ---
VAE_EPOCHS = 40
VAE_BATCH = 128
VAE_LR = 1e-3

MDNRNN_EPOCHS = 40
MDNRNN_BATCH = 64
MDNRNN_LR = 1e-3
SEQ_LEN = 64


class TransitionDataset(Dataset):
    """
    Dataset plat (une transition = un item) pour l'entraînement du VAE.

    Charge et concatène tous les fichiers episode_*.npz présents dans ``data_dir``.

    :param data_dir: Chemin vers le dossier de données.
    :type data_dir: str
    """

    def __init__(self, data_dir: str):
        files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
        obs_parts, next_obs_parts, action_parts = [], [], []
        for f in files:
            d = np.load(f)
            obs_parts.append(d["obs"])
            next_obs_parts.append(d["next_obs"])
            action_parts.append(d["actions"])
        self.obs = np.concatenate(obs_parts, axis=0)
        self.next_obs = np.concatenate(next_obs_parts, axis=0)
        self.actions = np.concatenate(action_parts, axis=0).astype(np.int64)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> tuple:
        return self.obs[idx], self.next_obs[idx], self.actions[idx]


class EpisodeDataset(Dataset):
    """
    Dataset séquentiel pour l'entraînement du MDN-RNN.

    Découpe chaque épisode en sous-séquences de longueur fixe ``seq_len``
    pour nourrir le LSTMCell pas à pas.

    :param data_dir: Chemin vers le dossier de données.
    :type data_dir: str
    :param seq_len: Longueur des sous-séquences (nombre de pas de temps).
    :type seq_len: int
    """

    def __init__(self, data_dir: str, seq_len: int = SEQ_LEN):
        files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
        self.sequences: list[tuple] = []
        for f in files:
            d = np.load(f)
            obs = d["obs"]
            next_obs = d["next_obs"]
            actions = d["actions"].astype(np.int64)
            rewards = d["rewards"].astype(np.float32)
            T = len(actions)
            for start in range(0, T - seq_len + 1, seq_len):
                self.sequences.append((
                    obs[start : start + seq_len],
                    next_obs[start : start + seq_len],
                    actions[start : start + seq_len],
                    rewards[start : start + seq_len],
                ))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        obs, next_obs, actions, rewards = self.sequences[idx]
        return obs, next_obs, actions, rewards


def train_vae(device: torch.device, writer: SummaryWriter) -> VAE:
    """
    Phase 1 : entraîne le VAE sur toutes les observations collectées.

    Minimise la loss ELBO = MSE reconstruction + KL divergence.
    Sauvegarde les poids dans ``checkpoints/vae.pt``.

    Métriques loggées dans TensorBoard :
      - ``VAE/loss_total``
      - ``VAE/loss_reconstruction``
      - ``VAE/loss_kl``

    :param device: Device d'entraînement (CPU ou CUDA).
    :type device: torch.device
    :param writer: SummaryWriter TensorBoard partagé.
    :type writer: SummaryWriter
    :returns: VAE entraîné et en mode ``eval()``.
    :rtype: VAE
    """
    print("\n=== Phase 1 : Entraînement du VAE ===")
    import torch.nn.functional as F

    dataset = TransitionDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=VAE_BATCH, shuffle=True,
                        num_workers=2, pin_memory=device.type == "cuda")

    vae = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=VAE_LR)

    t_start = time.time()
    for epoch in range(VAE_EPOCHS):
        t_epoch = time.time()
        vae.train()
        total_loss = total_recon = total_kl = 0.0

        for obs_batch, _, _ in loader:
            x = obs_array_to_tensor(obs_batch.numpy(), device)
            x_recon, mu, log_var = vae(x)

            recon = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
            kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss  = recon + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            total_recon += recon.item()
            total_kl    += kl.item()

        n = len(loader)
        writer.add_scalar("VAE/loss_total",          total_loss  / n, epoch)
        writer.add_scalar("VAE/loss_reconstruction", total_recon / n, epoch)
        writer.add_scalar("VAE/loss_kl",             total_kl    / n, epoch)

        elapsed   = time.time() - t_start
        avg_epoch = elapsed / (epoch + 1)
        remaining = avg_epoch * (VAE_EPOCHS - epoch - 1)
        eta       = f"{int(remaining // 3600):02d}:{int((remaining % 3600) // 60):02d}:{int(remaining % 60):02d}"
        write_status(2, "World Model — VAE", epoch + 1, VAE_EPOCHS,
                     {"loss": round(total_loss / n, 4),
                      "recon": round(total_recon / n, 4),
                      "kl": round(total_kl / n, 4)},
                     eta=eta)
        print(f"[VAE] Epoch {epoch + 1:3d}/{VAE_EPOCHS} | "
              f"loss={total_loss/n:.4f} | recon={total_recon/n:.4f} | kl={total_kl/n:.4f} | "
              f"{eta} restant")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(vae.state_dict(), os.path.join(CHECKPOINT_DIR, "vae.pt"))
    print(f"→ VAE sauvegardé dans {CHECKPOINT_DIR}/vae.pt")
    vae.eval()
    return vae


def train_mdn_rnn(vae: VAE, device: torch.device, writer: SummaryWriter) -> MDNRNN:
    """
    Phase 2 : entraîne le MDN-RNN sur des séquences de vecteurs latents.

    Le VAE est utilisé pour encoder les observations en vecteurs latents μ(x)
    (mode de la distribution, sans bruit) avant de nourrir le LSTM.
    Les poids du VAE sont gelés pendant cette phase.

    Minimise la NLL du mélange de gaussiennes sur z_{t+1}.
    Sauvegarde les poids dans ``checkpoints/mdn_rnn.pt``.

    Métriques loggées dans TensorBoard :
      - ``MDNRNN/loss_nll``

    :param vae: VAE pré-entraîné et gelé.
    :type vae: VAE
    :param device: Device d'entraînement.
    :type device: torch.device
    :param writer: SummaryWriter TensorBoard partagé.
    :type writer: SummaryWriter
    :returns: MDN-RNN entraîné et en mode ``eval()``.
    :rtype: MDNRNN
    """
    print("\n=== Phase 2 : Entraînement du MDN-RNN ===")
    dataset = EpisodeDataset(DATA_DIR, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=MDNRNN_BATCH, shuffle=True, num_workers=2)

    for p in vae.parameters():
        p.requires_grad_(False)

    mdn_rnn = MDNRNN(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        num_gaussians=NUM_GAUSSIANS,
    ).to(device)
    optimizer = torch.optim.Adam(mdn_rnn.parameters(), lr=MDNRNN_LR)

    t_start = time.time()
    for epoch in range(MDNRNN_EPOCHS):
        mdn_rnn.train()
        total_loss = 0.0

        for obs_seq, next_obs_seq, action_seq, reward_seq in loader:
            B, T = obs_seq.shape[:2]

            with torch.no_grad():
                z_flat, _ = vae.encode(
                    obs_array_to_tensor(obs_seq.numpy().reshape(B * T, 7, 7, 3), device)
                )
                z_next_flat, _ = vae.encode(
                    obs_array_to_tensor(next_obs_seq.numpy().reshape(B * T, 7, 7, 3), device)
                )

            z       = z_flat.view(B, T, LATENT_DIM)
            z_next  = z_next_flat.view(B, T, LATENT_DIM)
            actions = action_seq.to(device)
            rewards = reward_seq.to(device)    # (B, T)

            h, c = mdn_rnn.init_hidden(B, device)
            seq_loss = torch.tensor(0.0, device=device)

            for t in range(T):
                pi, mu, sigma, r_pred, h, c = mdn_rnn(z[:, t], actions[:, t], h, c)
                nll = MDNRNN.mdn_loss(pi, mu, sigma, z_next[:, t])
                r_loss = F.mse_loss(r_pred.squeeze(-1), rewards[:, t])
                seq_loss = seq_loss + nll + r_loss

            loss = seq_loss / T
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdn_rnn.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg      = total_loss / len(loader)
        elapsed  = time.time() - t_start
        avg_ep   = elapsed / (epoch + 1)
        remaining = avg_ep * (MDNRNN_EPOCHS - epoch - 1)
        eta      = f"{int(remaining // 3600):02d}:{int((remaining % 3600) // 60):02d}:{int(remaining % 60):02d}"
        writer.add_scalar("MDNRNN/loss_nll", avg, epoch)
        write_status(2, "World Model — MDN-RNN", epoch + 1, MDNRNN_EPOCHS,
                     {"nll": round(avg, 4)},
                     eta=eta)
        print(f"[MDN-RNN] Epoch {epoch + 1:3d}/{MDNRNN_EPOCHS} | "
              f"nll={avg:.4f} | {eta} restant")

    torch.save(mdn_rnn.state_dict(), os.path.join(CHECKPOINT_DIR, "mdn_rnn.pt"))
    print(f"→ MDN-RNN sauvegardé dans {CHECKPOINT_DIR}/mdn_rnn.pt")
    mdn_rnn.eval()
    return mdn_rnn


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device : {device} ({gpu_name})")

    if not glob.glob(os.path.join(DATA_DIR, "*.npz")):
        print(f"Erreur : aucun fichier .npz dans '{DATA_DIR}/'. Lancez 1_collect_data.py d'abord.")
        raise SystemExit(1)

    writer = SummaryWriter(log_dir=RUNS_DIR)
    print(f"TensorBoard → {RUNS_DIR}")

    vae = train_vae(device, writer)
    train_mdn_rnn(vae, device, writer)
    writer.close()
    print("\nWorld Model entraîné avec succès.")
