"""
Étape 1 — Collecte de données par jeu aléatoire.

Joue TARGET_TRANSITIONS pas dans l'environnement avec des actions uniformément
aléatoires et sauvegarde chaque épisode dans data/episode_XXXXXX.npz.

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
import random

import numpy as np
import requests

from utils import request_with_retry

SERVER_URL = "http://env_server:8000"
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
    session = requests.Session()

    total = 0
    episode_idx = 0

    print(f"Collecte de {TARGET_TRANSITIONS} transitions → {DATA_DIR}/")

    while total < TARGET_TRANSITIONS:
        res = request_with_retry(session, "GET", f"{SERVER_URL}/reset")
        obs = np.array(res.json()["obs"], dtype=np.uint8)

        obs_list, next_obs_list, action_list, reward_list, done_list = [], [], [], [], []

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = random.randint(0, 6)
            res = request_with_retry(session, "POST", f"{SERVER_URL}/step",
                                     json={"action": action})
            data = res.json()

            next_obs = np.array(data["obs"], dtype=np.uint8)
            obs_list.append(obs)
            next_obs_list.append(next_obs)
            action_list.append(action)
            reward_list.append(float(data["reward"]))
            done_list.append(bool(data["done"]))

            obs = next_obs
            total += 1

            if data["done"] or total >= TARGET_TRANSITIONS:
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

    print(f"\nCollecte terminée — {total} transitions dans {episode_idx} fichiers.")


if __name__ == "__main__":
    collect_data()
