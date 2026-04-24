import gymnasium as gym
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
env = gym.make("MiniGrid-Empty-8x8-v0")

class Action(BaseModel):
    action: int

@app.get("/reset")
def reset():
    obs, info = env.reset()
    # On convertit le numpy array en liste pour le JSON
    return {"obs": obs['image'].tolist()} 

@app.post("/step")
def step(act: Action):
    obs, reward, terminated, truncated, info = env.step(act.action)
    return {
        "obs": obs['image'].tolist(),
        "reward": float(reward),
        "done": bool(terminated or truncated)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    