import gymnasium as gym
import minigrid
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
env = gym.make("MiniGrid-Empty-8x8-v0")


class Action(BaseModel):
    """
    Modèle de données pour une action envoyée par le client.

    :param action: Indice de l'action à exécuter (0-6, selon l'espace d'actions MiniGrid).
    :type action: int
    """

    action: int


@app.get("/reset")
def reset() -> dict:
    """
    Réinitialise l'environnement MiniGrid et retourne l'observation initiale.

    Appelé en début d'épisode par le trainer. L'image est convertie de numpy array
    en liste Python pour la sérialisation JSON.

    :returns: Dictionnaire contenant l'observation initiale sous forme de liste 3D (7x7x3).
    :rtype: dict
    """
    obs, info = env.reset()
    return {"obs": obs['image'].tolist()}


@app.post("/step")
def step(act: Action) -> dict:
    """
    Exécute une action dans l'environnement et retourne le résultat.

    :param act: Action à exécuter, validée via le modèle Pydantic ``Action``.
    :type act: Action
    :returns: Dictionnaire contenant l'observation suivante (7x7x3), la récompense
              et un booléen indiquant si l'épisode est terminé.
    :rtype: dict
    """
    obs, reward, terminated, truncated, info = env.step(act.action)
    return {
        "obs": obs['image'].tolist(),
        "reward": float(reward),
        "done": bool(terminated or truncated)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
