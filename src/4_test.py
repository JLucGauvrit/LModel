"""
Test de l'agent World Model (V-M-C) sur MiniGrid.

Usage:
    python src/4_test.py [--episodes N] [--stochastic] [--env ENV_ID]

Sorties dans data/test/ :
    ep_XX.gif               — épisode animé annoté
    vae_reconstruction.png  — grille original vs reconstruction VAE
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym
import minigrid  # noqa: F401 — enregistre les envs MiniGrid
from PIL import Image, ImageDraw

from models import VAE, MDNRNN, Controller
from utils import obs_array_to_tensor, LATENT_DIM, HIDDEN_DIM, ACTION_DIM, NUM_GAUSSIANS

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "data/test"

ACTION_NAMES = ["turn_left", "turn_right", "forward", "pickup", "drop", "toggle", "done"]


# ---------------------------------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------------------------------

def load_models(device: torch.device) -> tuple[VAE, MDNRNN, Controller]:
    """
    Charge VAE, MDN-RNN et Controller depuis les checkpoints.

    :param device: Device PyTorch cible.
    :type device: torch.device
    :returns: Tuple (vae, mdn_rnn, controller) en mode eval.
    :rtype: tuple[VAE, MDNRNN, Controller]
    """
    def _load(cls, filename, **kwargs):
        model = cls(**kwargs).to(device)
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, filename), map_location=device))
        model.eval()
        return model

    vae = _load(VAE, "vae.pt", latent_dim=LATENT_DIM)
    mdn_rnn = _load(MDNRNN, "mdn_rnn.pt",
                    latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
                    hidden_dim=HIDDEN_DIM, num_gaussians=NUM_GAUSSIANS)
    controller = _load(Controller, "controller.pt",
                       latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_actions=ACTION_DIM)
    return vae, mdn_rnn, controller


# ---------------------------------------------------------------------------
# Annotation des frames
# ---------------------------------------------------------------------------

def _annotate(frame: np.ndarray, step: int, action: int, reward: float,
              total: float, pred_r: float, z_norm: float) -> Image.Image:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    W, H = img.size

    lines = [
        f"t={step:03d}  {ACTION_NAMES[action]}",
        f"r={reward:+.2f}  R={total:+.2f}",
        f"r^={pred_r:+.2f}  |z|={z_norm:.1f}",
    ]
    line_h = max(H // 20, 10)
    pad = 3

    for i, line in enumerate(lines):
        y = pad + i * (line_h + 1)
        draw.text((pad + 1, y + 1), line, fill=(0, 0, 0))
        draw.text((pad, y), line, fill=(255, 220, 0))

    return img


# ---------------------------------------------------------------------------
# Exécution d'un épisode
# ---------------------------------------------------------------------------

def run_episode(
    env: gym.Env,
    vae: VAE,
    mdn_rnn: MDNRNN,
    controller: Controller,
    device: torch.device,
    greedy: bool = True,
) -> dict:
    """
    Joue un épisode complet et renvoie frames annotées, log, métriques.

    :param env: Environnement Gymnasium déjà créé avec render_mode='rgb_array'.
    :type env: gym.Env
    :param vae: Modèle VAE en eval.
    :type vae: VAE
    :param mdn_rnn: Modèle MDN-RNN en eval.
    :type mdn_rnn: MDNRNN
    :param controller: Contrôleur en eval.
    :type controller: Controller
    :param device: Device PyTorch.
    :type device: torch.device
    :param greedy: Si True, argmax sur les logits ; sinon échantillonnage catégoriel.
    :type greedy: bool
    :returns: Dict avec 'frames', 'log', 'total_reward', 'steps', 'success', 'obs_samples'.
    :rtype: dict
    """
    gym_obs, _ = env.reset()
    obs = obs_array_to_tensor(gym_obs["image"][np.newaxis], device)
    h, c = mdn_rnn.init_hidden(1, device)

    frames: list[Image.Image] = []
    log: list[dict] = []
    obs_samples: list[np.ndarray] = []
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        raw_frame = env.render()

        with torch.no_grad():
            z, _ = vae.encode(obs)

            action_logits = controller(z, h)
            if greedy:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = torch.distributions.Categorical(logits=action_logits).sample()

            _, _, _, pred_reward, h, c = mdn_rnn(z, action, h, c)

        gym_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward

        entry = {
            "step": step,
            "action": action.item(),
            "reward": reward,
            "total_reward": total_reward,
            "pred_reward": pred_reward.item(),
            "z_norm": z.norm().item(),
        }
        log.append(entry)
        frames.append(_annotate(raw_frame, **{k: entry[k] for k in
                                              ("step", "action", "reward", "total_reward",
                                               "pred_reward", "z_norm")}))

        if step % 4 == 0 and len(obs_samples) < 6:
            obs_samples.append(gym_obs["image"].copy())

        if not done:
            obs = obs_array_to_tensor(gym_obs["image"][np.newaxis], device)

        step += 1

    frames.append(Image.fromarray(env.render()))

    return {
        "frames": frames,
        "log": log,
        "total_reward": total_reward,
        "steps": step,
        "success": total_reward > 0,
        "obs_samples": obs_samples,
    }


# ---------------------------------------------------------------------------
# Affichage console
# ---------------------------------------------------------------------------

def print_episode(ep: int, result: dict) -> None:
    status = "SUCCESS" if result["success"] else "FAIL"
    print(f"\n{'─'*56}")
    print(f" Episode {ep:02d} — {status:7s} | "
          f"Steps: {result['steps']:3d} | Reward: {result['total_reward']:+.3f}")
    print(f" {'t':>4}  {'action':<11}  {'r':>6}  {'R_tot':>6}  {'r_pred':>7}  {'|z|':>5}")
    print(f" {'─'*50}")
    for e in result["log"]:
        print(f" {e['step']:>4}  {ACTION_NAMES[e['action']]:<11}  "
              f"{e['reward']:>+6.3f}  {e['total_reward']:>+6.3f}  "
              f"{e['pred_reward']:>+7.3f}  {e['z_norm']:>5.2f}")


# ---------------------------------------------------------------------------
# Grille de reconstruction VAE
# ---------------------------------------------------------------------------

def save_vae_reconstruction(vae: VAE, device: torch.device,
                             obs_list: list[np.ndarray], path: str) -> None:
    """
    Sauvegarde une grille PNG original / reconstruction VAE.

    :param vae: Modèle VAE en eval.
    :type vae: VAE
    :param device: Device PyTorch.
    :type device: torch.device
    :param obs_list: Liste d'observations brutes (H, W, C) shape (7, 7, 3).
    :type obs_list: list[np.ndarray]
    :param path: Chemin de sortie du PNG.
    :type path: str
    """
    n = len(obs_list)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, obs_arr in enumerate(obs_list):
        x = obs_array_to_tensor(obs_arr[np.newaxis], device)
        with torch.no_grad():
            x_recon, _, _ = vae(x)

        orig = (obs_arr.astype(np.float32) / 10.0).clip(0, 1)
        recon = x_recon[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

        axes[0, i].imshow(orig)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon)
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original", fontsize=8, loc="left")
    axes[1, 0].set_title("VAE recon", fontsize=8, loc="left")

    plt.tight_layout(pad=0.5)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVAE reconstruction : {path}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def test_agent(episodes: int = 5, greedy: bool = True, env_id: str = "MiniGrid-Empty-8x8-v0") -> None:
    """
    Lance N épisodes de test, sauvegarde les GIFs et la grille VAE.

    :param episodes: Nombre d'épisodes.
    :type episodes: int
    :param greedy: Politique argmax (True) ou stochastique (False).
    :type greedy: bool
    :param env_id: ID de l'environnement Gymnasium.
    :type env_id: str
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mode = "greedy" if greedy else "stochastic"
    print(f"Device: {device} | Mode: {mode} | Env: {env_id}")

    vae, mdn_rnn, controller = load_models(device)
    env = gym.make(env_id, render_mode="rgb_array")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    all_obs_samples: list[np.ndarray] = []

    for ep in range(episodes):
        result = run_episode(env, vae, mdn_rnn, controller, device, greedy=greedy)
        print_episode(ep, result)

        gif_path = os.path.join(OUTPUT_DIR, f"ep_{ep:02d}.gif")
        result["frames"][0].save(
            gif_path, save_all=True, append_images=result["frames"][1:],
            duration=150, loop=0,
        )
        print(f"  → {gif_path}")

        results.append(result)
        all_obs_samples.extend(result["obs_samples"][:2])

    env.close()

    if all_obs_samples:
        save_vae_reconstruction(vae, device, all_obs_samples[:8],
                                os.path.join(OUTPUT_DIR, "vae_reconstruction.png"))

    # Tableau récapitulatif
    print(f"\n{'═'*56}")
    print(f" RÉSUMÉ — {episodes} épisodes ({mode})")
    print(f" {'Ep':>3}  {'Steps':>5}  {'Reward':>8}  {'Résultat':>9}")
    print(f" {'─'*38}")
    for i, r in enumerate(results):
        status = "SUCCESS" if r["success"] else "fail"
        print(f" {i:>3}  {r['steps']:>5}  {r['total_reward']:>+8.3f}  {status:>9}")

    successes = sum(r["success"] for r in results)
    avg_r = np.mean([r["total_reward"] for r in results])
    avg_s = np.mean([r["steps"] for r in results])
    print(f"\n Taux de succès : {successes}/{episodes} ({100*successes/episodes:.0f}%)")
    print(f" Reward moyen   : {avg_r:+.3f}  |  Steps moyens : {avg_s:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de l'agent World Model")
    parser.add_argument("--episodes", type=int, default=5, help="Nombre d'épisodes (défaut: 5)")
    parser.add_argument("--stochastic", action="store_true", help="Politique stochastique (défaut: greedy)")
    parser.add_argument("--env", default="MiniGrid-Empty-8x8-v0", help="ID de l'environnement")
    args = parser.parse_args()

    test_agent(episodes=args.episodes, greedy=not args.stochastic, env_id=args.env)
