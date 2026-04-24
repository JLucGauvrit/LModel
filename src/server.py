import io

import gymnasium as gym
import minigrid
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel
import uvicorn

app = FastAPI()
env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
env.reset()  # Initialise le rendu dès le démarrage

# ---------------------------------------------------------------------------
# Dashboard HTML — servi à http://localhost:8000/ui
# ---------------------------------------------------------------------------
_UI_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>MiniGrid — Live Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d1117; color: #c9d1d9;
      font-family: 'Courier New', Courier, monospace;
      display: flex; flex-direction: column; align-items: center;
      min-height: 100vh; padding: 2rem 1rem; gap: 1.5rem;
    }
    header { text-align: center; }
    header h1 { color: #58a6ff; font-size: 1.4rem; letter-spacing: 3px; text-transform: uppercase; }
    header p  { color: #6e7681; font-size: 0.8rem; margin-top: 0.3rem; }
    #frame-container {
      position: relative; border: 1px solid #30363d;
      border-radius: 6px; overflow: hidden; line-height: 0;
    }
    #frame {
      image-rendering: pixelated; image-rendering: crisp-edges;
      width: 480px; height: 480px; display: block;
    }
    #overlay {
      position: absolute; bottom: 0; left: 0; right: 0;
      background: rgba(0,0,0,0.65); padding: 0.4rem 0.8rem;
      font-size: 0.75rem; color: #8b949e;
      display: flex; justify-content: space-between;
    }
    #dot {
      display: inline-block; width: 7px; height: 7px;
      border-radius: 50%; background: #3fb950; margin-right: 5px;
      animation: pulse 1.5s ease-in-out infinite;
    }
    #dot.err { background: #f85149; animation: none; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
    .links { font-size: 0.8rem; color: #6e7681; }
    .links a { color: #58a6ff; text-decoration: none; }
    .links a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <header>
    <h1>&#9889; MiniGrid — Live View</h1>
    <p>World Model FPGA &mdash; Prototype Grid Routing</p>
  </header>
  <div id="frame-container">
    <img id="frame" src="/render" alt="game" />
    <div id="overlay">
      <span><span id="dot"></span><span id="st">Connexion...</span></span>
      <span id="fps">— fps</span>
    </div>
  </div>
  <div class="links">
    Métriques d'entraînement &rarr;
    <a href="http://localhost:6006" target="_blank">TensorBoard :6006</a>
  </div>
  <script>
    const img = document.getElementById('frame');
    const dot = document.getElementById('dot');
    const st  = document.getElementById('st');
    const fps = document.getElementById('fps');
    let frames = 0, last = performance.now();
    function tick() {
      const n = new Image();
      n.onload = () => {
        img.src = n.src; frames++;
        const now = performance.now();
        if (now - last >= 1000) {
          fps.textContent = frames + ' fps'; frames = 0; last = now;
        }
        dot.classList.remove('err');
        st.textContent = new Date().toLocaleTimeString('fr-FR');
      };
      n.onerror = () => { dot.classList.add('err'); st.textContent = 'Hors ligne'; fps.textContent = '0 fps'; };
      n.src = '/render?t=' + Date.now();
    }
    setInterval(tick, 200);
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Modèle de données
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Corps de la requête POST /step.

    :param action: Indice de l'action (0–6, espace d'actions MiniGrid).
    :type action: int
    """

    action: int


# ---------------------------------------------------------------------------
# Endpoints API
# ---------------------------------------------------------------------------

@app.get("/reset")
def reset() -> dict:
    """
    Réinitialise l'environnement et retourne l'observation initiale.

    :returns: ``{ "obs": [[[int]]] }`` — image 7×7×3 sérialisée.
    :rtype: dict
    """
    obs, _ = env.reset()
    return {"obs": obs["image"].tolist()}


@app.post("/step")
def step(act: Action) -> dict:
    """
    Exécute une action et retourne la transition.

    :param act: Action validée par Pydantic.
    :type act: Action
    :returns: ``{ "obs", "reward", "done" }``.
    :rtype: dict
    """
    obs, reward, terminated, truncated, _ = env.step(act.action)
    return {
        "obs": obs["image"].tolist(),
        "reward": float(reward),
        "done": bool(terminated or truncated),
    }


@app.get("/render")
def render():
    """
    Retourne le frame courant de l'environnement comme image PNG.

    Utilisé par le dashboard ``/ui`` pour afficher le jeu en temps réel.
    La réponse ne doit pas être mise en cache (``Cache-Control: no-store``).

    :returns: Image PNG du frame courant.
    :rtype: StreamingResponse
    """
    frame = env.render()
    if frame is None:
        frame = np.full((256, 256, 3), 60, dtype=np.uint8)
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    """
    Sert le dashboard HTML de visualisation live du jeu.

    Affiche le frame courant rafraîchi toutes les 200 ms (~5 fps)
    et un lien vers TensorBoard pour les métriques d'entraînement.

    :returns: Page HTML complète.
    :rtype: str
    """
    return _UI_HTML


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
