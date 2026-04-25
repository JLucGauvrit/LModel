"""
Étape 1 — Collecte de données par jeu aléatoire.

Joue TARGET_TRANSITIONS pas dans l'environnement MiniGrid instancié localement
avec des actions uniformément aléatoires et sauvegarde chaque épisode dans data/episode_XXXXXX.npz.

Chaque fichier contient les arrays :
  obs      (T, 7, 7, 3) uint8  — observation à l'instant t
  next_obs (T, 7, 7, 3) uint8  — observation à l'instant t+1
  actions  (T,)          int64  — action choisie
  rewards  (T,)          float32 — récompense obtenue
  dones    (T,)          bool   — fin d'épisode

Usage :
  python src/1_collect_data.py
"""

import os

import gymnasium as gym
import minigrid  # noqa: F401 — enregistre les environnements MiniGrid
import numpy as np

DATA_DIR = "data"
TARGET_TRANSITIONS = 10_000
MAX_STEPS_PER_EPISODE = 500


def collect_data() -> None:
    """
    Boucle principale de collecte : joue aléatoirement jusqu'à TARGET_TRANSITIONS transitions.

    Les épisodes sont sauvegardés au fil de l'eau dans DATA_DIR pour éviter
    toute perte en cas d'interruption.

    :returns: None
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    env = gym.make("MiniGrid-Empty-8x8-v0")

    total = 0
    episode_idx = 0

    print(f"Collecte de {TARGET_TRANSITIONS} transitions → {DATA_DIR}/")

    while total < TARGET_TRANSITIONS:
        gym_obs, _ = env.reset()
        obs = gym_obs["image"]  # (7, 7, 3) uint8

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

            if done or total >= TARGET_TRANSITIONS:
                break

        np.savez_compressed(
            os.path.join(DATA_DIR, f"episode_{episode_idx:06d}.npz"),
            obs=np.array(obs_list, dtype=np.uint8),
            next_obs=np.array(next_obs_list, dtype=np.uint8),
            actions=np.array(action_list, dtype=np.int64),
            rewards=np.array(reward_list, dtype=np.float32),
            dones=np.array(done_list, dtype=bool),
        )

        episode_idx += 1
        print(f"  Épisode {episode_idx:4d} | {len(action_list):3d} steps | "
              f"Total : {total}/{TARGET_TRANSITIONS}")

    env.close()
    print(f"\nCollecte terminée — {total} transitions dans {episode_idx} fichiers.")


if __name__ == "__main__":
    collect_data()
