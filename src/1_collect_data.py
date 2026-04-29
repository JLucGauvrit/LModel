"""
Étape 1 — Collecte de données par jeu aléatoire sur plusieurs environnements MiniGrid.

Pour chaque environnement de ENVS, joue TRANSITIONS_PER_ENV pas avec des actions
uniformément aléatoires et sauvegarde chaque épisode dans data/episode_XXXXXX.npz.

Tous les environnements MiniGrid partagent la même observation partielle 7×7×3 et
le même espace d'actions à 7 actions, ce qui les rend directement compatibles avec
le pipeline VAE / MDN-RNN / Controller.

Chaque fichier contient les arrays :
  obs        (T, 7, 7, 3) uint8   — observation à l'instant t
  next_obs   (T, 7, 7, 3) uint8   — observation à l'instant t+1
  actions    (T,)          int64   — action choisie
  rewards    (T,)          float32 — récompense obtenue
  dones      (T,)          bool    — fin d'épisode (terminated OR truncated)
  terminated (T,)          bool    — but atteint (pas timeout)
  env_id     ()            int64   — index de l'environnement dans ENVS

Usage :
  python src/1_collect_data.py
"""

import os
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401 — enregistre les environnements MiniGrid
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from utils import write_status  # charge aussi load_dotenv()

DATA_DIR = "data"


class DistanceRewardWrapper(gym.Wrapper):
    """
    Ajoute une récompense dense basée sur le potentiel de distance Manhattan
    vers le but : r_shaping = (d_prev - d_new) * SHAPING_SCALE.

    Positif quand l'agent se rapproche, négatif quand il s'éloigne.
    Ne remplace pas la récompense terminale (reward != 0).

    :param env: Environnement MiniGrid à envelopper.
    :type env: gym.Env
    """

    SHAPING_SCALE: float = 0.1

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._goal_pos: tuple[int, int] | None = None
        self._prev_dist: float = 0.0

    def reset(self, **kwargs: Any) -> tuple:
        obs, info = self.env.reset(**kwargs)
        self._goal_pos = self._find_goal()
        self._prev_dist = self._manhattan(self.env.unwrapped.agent_pos)
        return obs, info

    def step(self, action: int) -> tuple:
        obs, reward, term, trunc, info = self.env.step(action)
        if reward == 0 and self._goal_pos is not None:
            new_dist = self._manhattan(self.env.unwrapped.agent_pos)
            reward = (self._prev_dist - new_dist) * self.SHAPING_SCALE
            self._prev_dist = new_dist
        return obs, reward, term, trunc, info

    def _find_goal(self) -> tuple[int, int] | None:
        u = self.env.unwrapped
        if hasattr(u, "goal_pos"):
            return u.goal_pos
        for i in range(u.grid.width):
            for j in range(u.grid.height):
                cell = u.grid.get(i, j)
                if cell is not None and cell.type == "goal":
                    return (i, j)
        return None

    def _manhattan(self, pos: tuple[int, int]) -> float:
        if self._goal_pos is None:
            return 0.0
        return float(abs(pos[0] - self._goal_pos[0]) + abs(pos[1] - self._goal_pos[1]))

class FullObsResizeWrapper(gym.ObservationWrapper):
    """Pads FullyObsWrapper full-grid image to (7, 7, 3) for VAE compatibility.

    MiniGrid-Empty-5x5-v0 has a 5×5 total grid (with walls), so FullyObsWrapper
    gives (5, 5, 3). We zero-pad to the VAE's expected input size (7, 7, 3).
    Zero cells are interpreted by the VAE as "outside grid boundary".
    """

    TARGET_H = TARGET_W = 7

    def observation(self, obs: dict) -> dict:
        img = obs["image"]  # (H, W, 3)
        H, W, C = img.shape
        if H == self.TARGET_H and W == self.TARGET_W:
            return obs
        padded = np.zeros((self.TARGET_H, self.TARGET_W, C), dtype=img.dtype)
        padded[:H, :W, :] = img
        return {**obs, "image": padded}


# ── Hyperparamètres — surchargeables via .env ─────────────────────────────────
MAX_STEPS_PER_EPISODE: int = int(os.getenv("MAX_STEPS_PER_EPISODE", 500))
TARGET_TRANSITIONS:    int = int(os.getenv("TARGET_TRANSITIONS", 200_000))

_default_envs = (
    "MiniGrid-Empty-5x5-v0,"
    "MiniGrid-Empty-6x6-v0,"
    "MiniGrid-DoorKey-5x5-v0,"
    "MiniGrid-SimpleCrossingS9N1-v0"
)
ENVS: list[str] = os.getenv("COLLECT_ENVS", _default_envs).split(",")

TRANSITIONS_PER_ENV: int = TARGET_TRANSITIONS // len(ENVS)
GREEDY_EPSILON: float = float(os.getenv("GREEDY_EPSILON", 0.3))


def greedy_action(env) -> int:
    """Returns the greedy action that moves the agent toward the goal tile.

    :param env: The wrapped MiniGrid environment.
    :returns: Integer action index.
    :rtype: int
    """
    agent_pos = env.unwrapped.agent_pos   # (x, y) in grid coords
    agent_dir = env.unwrapped.agent_dir   # 0=right, 1=down, 2=left, 3=up

    goal_pos = None
    for x in range(env.unwrapped.width):
        for y in range(env.unwrapped.height):
            cell = env.unwrapped.grid.get(x, y)
            if cell is not None and cell.type == "goal":
                goal_pos = (x, y)
                break
        if goal_pos:
            break

    if goal_pos is None:
        return env.action_space.sample()

    dx = goal_pos[0] - agent_pos[0]
    dy = goal_pos[1] - agent_pos[1]

    if dx == 0 and dy == 0:
        return 2  # already adjacent — move forward to trigger termination

    # Desired direction: choose axis with larger delta
    # MiniGrid dirs: 0=+x(right), 1=+y(down), 2=-x(left), 3=-y(up)
    if abs(dx) >= abs(dy):
        desired_dir = 0 if dx > 0 else 2
    else:
        desired_dir = 1 if dy > 0 else 3

    if agent_dir == desired_dir:
        return 2  # forward
    if (desired_dir - agent_dir) % 4 == 1:
        return 1  # turn right (1 clockwise step)
    return 0      # turn left


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
    env = DistanceRewardWrapper(FullObsResizeWrapper(FullyObsWrapper(gym.make(env_name))))
    total = 0
    episode_idx = episode_offset
    global_offset = env_id * TRANSITIONS_PER_ENV

    print(f"\n[{env_name}] Collecte de {transitions} transitions…")

    while total < transitions:
        gym_obs, _ = env.reset()
        obs = gym_obs["image"]

        obs_list, next_obs_list, action_list, reward_list, done_list, terminated_list = [], [], [], [], [], []

        for _ in range(MAX_STEPS_PER_EPISODE):
            if np.random.rand() < GREEDY_EPSILON:
                action = greedy_action(env)
            else:
                action = env.action_space.sample()
            gym_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = gym_obs["image"]
            done = terminated or truncated

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            action_list.append(action)
            reward_list.append(float(reward))
            done_list.append(done)
            terminated_list.append(bool(terminated))   # NEW — goal reached, not timeout

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
            terminated=np.array(terminated_list, dtype=bool),   # NEW
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
