"""
Étape 1 — Collecte de données par jeu aléatoire sur plusieurs environnements MiniGrid.

Pour chaque environnement de ENVS, joue TRANSITIONS_PER_ENV pas avec des actions
uniformément aléatoires et sauvegarde chaque épisode dans data/episode_XXXXXX.npz.

Tous les environnements MiniGrid partagent la même observation partielle 7×7×3 et
le même espace d'actions à 7 actions, ce qui les rend directement compatibles avec
le pipeline VAE / MDN-RNN / Controller.

Chaque fichier contient les arrays :
  obs      (T, 7, 7, 3) uint8  — observation à l'instant t
  next_obs (T, 7, 7, 3) uint8  — observation à l'instant t+1
  actions  (T,)          int64  — action choisie
  rewards  (T,)          float32 — récompense obtenue
  dones    (T,)          bool   — fin d'épisode
  env_id   ()            int64  — index de l'environnement dans ENVS

Usage :
  python src/1_collect_data.py
"""

import os

import gymnasium as gym
import minigrid  # noqa: F401 — enregistre les environnements MiniGrid
import numpy as np

from utils import write_status  # charge aussi load_dotenv()

DATA_DIR = "data"

# ── Hyperparamètres — surchargeables via .env ─────────────────────────────────
MAX_STEPS_PER_EPISODE: int = int(os.getenv("MAX_STEPS_PER_EPISODE", 500))
TARGET_TRANSITIONS:    int = int(os.getenv("TARGET_TRANSITIONS", 200_000))

_default_envs = (
    "MiniGrid-Empty-8x8-v0,"
    "MiniGrid-FourRooms-v0,"
    "MiniGrid-DoorKey-5x5-v0,"
    "MiniGrid-SimpleCrossingS9N1-v0"
)
ENVS: list[str] = os.getenv("COLLECT_ENVS", _default_envs).split(",")

TRANSITIONS_PER_ENV: int = TARGET_TRANSITIONS // len(ENVS)


def collect_env(env_name: str, env_id: int, transitions: int, episode_offset: int) -> int:
    """
    Collecte ``transitions`` transitions pour un environnement donné.

    Les épisodes sont sauvegardés au fil de l'eau dans DATA_DIR pour éviter
    toute perte en cas d'interruption.

    :param env_name: Identifiant gymnasium de l'environnement.
    :type env_name: str
    :param env_id: Index numérique de l'environnement dans ENVS.
    :type env_id: int
    :param transitions: Nombre de transitions à collecter.
    :type transitions: int
    :param episode_offset: Index du premier épisode pour nommer les fichiers.
    :type episode_offset: int
    :returns: Nombre d'épisodes sauvegardés.
    :rtype: int
    """
    env = gym.make(env_name)
    total = 0
    episode_idx = episode_offset
    global_offset = env_id * TRANSITIONS_PER_ENV

    print(f"\n[{env_name}] Collecte de {transitions} transitions…")

    while total < transitions:
        gym_obs, _ = env.reset()
        obs = gym_obs["image"]

        obs_list, next_obs_list, action_list, reward_list, done_list = [], [], [], [], []

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = env.action_space.sample()
            gym_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = gym_obs["image"]
            done = terminated or truncated

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            action_list.append(action)
            reward_list.append(float(reward))
            done_list.append(done)

            obs = next_obs
            total += 1

            if done or total >= transitions:
                break

        np.savez_compressed(
            os.path.join(DATA_DIR, f"episode_{episode_idx:06d}.npz"),
            obs=np.array(obs_list, dtype=np.uint8),
            next_obs=np.array(next_obs_list, dtype=np.uint8),
            actions=np.array(action_list, dtype=np.int64),
            rewards=np.array(reward_list, dtype=np.float32),
            dones=np.array(done_list, dtype=bool),
            env_id=np.int64(env_id),
        )

        episode_idx += 1
        write_status(
            1,
            f"Collecte — {env_name}",
            global_offset + total,
            TARGET_TRANSITIONS,
        )
        print(f"  Épisode {episode_idx - episode_offset:4d} | "
              f"{len(action_list):3d} steps | "
              f"Env {total}/{transitions} | "
              f"Total {global_offset + total}/{TARGET_TRANSITIONS}")

    env.close()
    return episode_idx - episode_offset


def collect_data() -> None:
    """
    Boucle principale de collecte : itère sur ENVS et collecte TRANSITIONS_PER_ENV
    transitions par environnement.

    :returns: None
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    episode_offset = 0

    print(f"Collecte de {TARGET_TRANSITIONS} transitions sur {len(ENVS)} environnements")
    print(f"  {TRANSITIONS_PER_ENV} transitions par env → {DATA_DIR}/\n")

    for env_id, env_name in enumerate(ENVS):
        n_episodes = collect_env(env_name, env_id, TRANSITIONS_PER_ENV, episode_offset)
        episode_offset += n_episodes
        print(f"→ {env_name} terminé : {n_episodes} épisodes")

    print(f"\nCollecte terminée — {TARGET_TRANSITIONS} transitions dans {episode_offset} fichiers.")


if __name__ == "__main__":
    collect_data()
