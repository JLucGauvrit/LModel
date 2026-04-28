"""
UI web pour l'inférence de l'agent World Model V-M-C.

Endpoints :
  GET  /           → page HTML
  POST /run        → lance un run (JSON: {episodes, greedy, env_id})
  GET  /status     → état courant pour le polling
  GET  /media/{f}  → fichiers GIF / PNG générés
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import gymnasium as gym
import minigrid  # noqa: F401
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from models import VAE, MDNRNN, Controller
from utils import obs_array_to_tensor, LATENT_DIM, HIDDEN_DIM, ACTION_DIM, NUM_GAUSSIANS

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("data/test")
ACTION_NAMES = ["turn_left", "turn_right", "forward", "pickup", "drop", "toggle", "done"]

AVAILABLE_ENVS = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-Empty-16x16-v0",
    "MiniGrid-FourRooms-v0",
    "MiniGrid-DoorKey-5x5-v0",
]

# ---------------------------------------------------------------------------
# État global partagé
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_state: dict = {
    "status": "idle",       # idle | running | done | error
    "config": {},
    "progress": {"ep": 0, "total": 0, "step": 0},
    "log": [],              # entrées {ep, step, action, reward, total_reward, pred_reward, z_norm}
    "episodes": [],         # résumés {ep, steps, total_reward, success}
    "media": [],            # noms de fichiers disponibles dans OUTPUT_DIR
    "error": None,
}

# Chargement des modèles une seule fois au démarrage
_models: dict | None = None
_device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_models() -> dict:
    def _load(cls, filename, **kwargs):
        m = cls(**kwargs).to(_device)
        m.load_state_dict(torch.load(CHECKPOINT_DIR / filename, map_location=_device))
        m.eval()
        return m

    return {
        "vae": _load(VAE, "vae.pt", latent_dim=LATENT_DIM),
        "mdn_rnn": _load(MDNRNN, "mdn_rnn.pt",
                         latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
                         hidden_dim=HIDDEN_DIM, num_gaussians=NUM_GAUSSIANS),
        "controller": _load(Controller, "controller.pt",
                             latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_actions=ACTION_DIM),
    }


# ---------------------------------------------------------------------------
# Inférence (tourne dans un thread séparé)
# ---------------------------------------------------------------------------

def _run_inference(episodes: int, greedy: bool, env_id: str) -> None:
    global _models

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with _lock:
        _state["status"] = "running"
        _state["progress"] = {"ep": 0, "total": episodes, "step": 0}
        _state["log"] = []
        _state["episodes"] = []
        _state["media"] = []
        _state["error"] = None

    try:
        if _models is None:
            _models = _load_models()

        vae = _models["vae"]
        mdn_rnn = _models["mdn_rnn"]
        controller = _models["controller"]

        env = gym.make(env_id, render_mode="rgb_array")

        for ep in range(episodes):
            with _lock:
                _state["progress"]["ep"] = ep
                _state["progress"]["step"] = 0

            frames, log_ep = _run_episode(env, vae, mdn_rnn, controller, greedy, ep)

            # Sauvegarde GIF
            gif_name = f"ep_{ep:02d}.gif"
            gif_path = OUTPUT_DIR / gif_name
            frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                           duration=150, loop=0)

            ep_log = log_ep[-1] if log_ep else {}
            success = ep_log.get("total_reward", 0) > 0

            with _lock:
                _state["episodes"].append({
                    "ep": ep,
                    "steps": ep_log.get("step", 0) + 1,
                    "total_reward": ep_log.get("total_reward", 0.0),
                    "success": success,
                })
                _state["media"].append(gif_name)

        env.close()

        # Grille de reconstruction VAE
        obs_samples = _collect_obs_samples(env_id, vae)
        if obs_samples:
            _save_vae_recon(vae, obs_samples)
            with _lock:
                _state["media"].append("vae_reconstruction.png")

        with _lock:
            _state["status"] = "done"

    except Exception as exc:
        with _lock:
            _state["status"] = "error"
            _state["error"] = str(exc)
        raise


def _run_episode(env, vae, mdn_rnn, controller, greedy: bool, ep: int):
    from PIL import Image, ImageDraw

    gym_obs, _ = env.reset()
    obs = obs_array_to_tensor(gym_obs["image"][np.newaxis], _device)
    h, c = mdn_rnn.init_hidden(1, _device)

    frames = []
    log_ep = []
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        raw_frame = env.render()

        with torch.no_grad():
            z, _ = vae.encode(obs)
            action_logits = controller(z, h)
            action = (torch.argmax(action_logits, dim=-1) if greedy
                      else torch.distributions.Categorical(logits=action_logits).sample())
            _, _, _, pred_reward, h, c = mdn_rnn(z, action, h, c)

        gym_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward

        entry = {
            "ep": ep, "step": step,
            "action": action.item(), "action_name": ACTION_NAMES[action.item()],
            "reward": round(float(reward), 4),
            "total_reward": round(float(total_reward), 4),
            "pred_reward": round(float(pred_reward.item()), 4),
            "z_norm": round(float(z.norm().item()), 3),
        }
        log_ep.append(entry)

        with _lock:
            _state["log"].append(entry)
            _state["progress"]["step"] = step

        frames.append(_annotate(raw_frame, entry))

        if not done:
            obs = obs_array_to_tensor(gym_obs["image"][np.newaxis], _device)
        step += 1

    frames.append(Image.fromarray(env.render()))
    return frames, log_ep


def _annotate(frame, entry: dict):
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    W, H = img.size
    lh = max(H // 20, 9)
    pad = 3
    lines = [
        f"t={entry['step']:03d}  {entry['action_name']}",
        f"r={entry['reward']:+.2f}  R={entry['total_reward']:+.2f}",
        f"r^={entry['pred_reward']:+.2f}  |z|={entry['z_norm']:.1f}",
    ]
    for i, line in enumerate(lines):
        y = pad + i * (lh + 1)
        draw.text((pad + 1, y + 1), line, fill=(0, 0, 0))
        draw.text((pad, y), line, fill=(255, 220, 0))
    return img


def _collect_obs_samples(env_id: str, vae) -> list:
    try:
        env = gym.make(env_id, render_mode="rgb_array")
        obs, _ = env.reset()
        samples = [obs["image"].copy()]
        for _ in range(7):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            samples.append(obs["image"].copy())
            if term or trunc:
                break
        env.close()
        return samples[:6]
    except Exception:
        return []


def _save_vae_recon(vae, obs_list: list) -> None:
    import matplotlib.pyplot as plt

    n = len(obs_list)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, obs_arr in enumerate(obs_list):
        x = obs_array_to_tensor(obs_arr[np.newaxis], _device)
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
    plt.savefig(OUTPUT_DIR / "vae_reconstruction.png", dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# API FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="V-M-C Inference UI")


class RunConfig(BaseModel):
    """
    Paramètres d'un run d'inférence.

    :param episodes: Nombre d'épisodes à jouer.
    :type episodes: int
    :param greedy: Si True, l'agent choisit l'action de plus haute probabilité.
    :type greedy: bool
    :param env_id: Identifiant de l'environnement Gymnasium.
    :type env_id: str
    """

    episodes: int = 5
    greedy: bool = True
    env_id: str = "MiniGrid-Empty-8x8-v0"


@app.post("/run")
def start_run(cfg: RunConfig) -> JSONResponse:
    """
    Lance un run d'inférence dans un thread démon.

    Refuse si un run est déjà en cours (HTTP 409).

    :param cfg: Configuration du run (épisodes, politique, environnement).
    :type cfg: RunConfig
    :returns: ``{"started": true}`` si le thread a démarré.
    :rtype: JSONResponse
    :raises HTTPException: 409 si un run est déjà en cours.
    """
    with _lock:
        if _state["status"] == "running":
            raise HTTPException(status_code=409, detail="Un run est déjà en cours.")
        _state["config"] = cfg.model_dump()

    t = threading.Thread(target=_run_inference,
                         args=(cfg.episodes, cfg.greedy, cfg.env_id), daemon=True)
    t.start()
    return JSONResponse({"started": True})


@app.get("/status")
def get_status() -> JSONResponse:
    """
    Retourne l'état courant du run d'inférence.

    :returns: Snapshot thread-safe de ``_state`` : status, config, progress,
              log, episodes, media, error.
    :rtype: JSONResponse
    """
    with _lock:
        return JSONResponse(dict(_state))


@app.get("/media/{filename}")
def get_media(filename: str) -> FileResponse:
    """
    Sert un fichier GIF ou PNG généré par le run d'inférence.

    :param filename: Nom du fichier dans ``data/test/``.
    :type filename: str
    :returns: Fichier image (``image/gif`` ou ``image/png``).
    :rtype: FileResponse
    :raises HTTPException: 404 si le fichier n'existe pas.
    """
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404)
    media_type = "image/gif" if filename.endswith(".gif") else "image/png"
    return FileResponse(path, media_type=media_type)


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    """
    Sert l'interface HTML de l'UI d'inférence.

    :returns: Page HTML complète.
    :rtype: str
    """
    return _HTML


# ---------------------------------------------------------------------------
# HTML / CSS / JS
# ---------------------------------------------------------------------------

_ENV_OPTIONS = "\n".join(
    f'<option value="{e}"{" selected" if e == "MiniGrid-Empty-8x8-v0" else ""}>{e}</option>'
    for e in AVAILABLE_ENVS
)

_HTML = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>V-M-C Inference</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0d1117; color: #c9d1d9;
      font-family: 'Courier New', Courier, monospace;
      min-height: 100vh; padding: 2rem 1rem;
    }}
    h1 {{ color: #58a6ff; font-size: 1.3rem; letter-spacing: 3px;
          text-transform: uppercase; margin-bottom: .3rem; }}
    header p {{ color: #6e7681; font-size: .8rem; margin-bottom: 2rem; }}

    .layout {{ display: grid; grid-template-columns: 280px 1fr; gap: 1.2rem;
               max-width: 1100px; margin: 0 auto; }}

    .card {{ background: #161b22; border: 1px solid #30363d;
             border-radius: 8px; padding: 1.2rem 1.4rem; }}
    .card-title {{ color: #58a6ff; font-size: .8rem; letter-spacing: 2px;
                   text-transform: uppercase; margin-bottom: 1rem; }}

    label {{ display: block; color: #8b949e; font-size: .78rem;
             margin-top: .8rem; margin-bottom: .3rem; }}
    input[type=number], select {{
      width: 100%; background: #0d1117; color: #c9d1d9;
      border: 1px solid #30363d; border-radius: 4px;
      padding: .4rem .6rem; font-family: inherit; font-size: .88rem;
    }}
    .radio-group {{ display: flex; gap: .6rem; margin-top: .3rem; }}
    .radio-group label {{
      margin: 0; flex: 1; text-align: center;
      padding: .4rem; border: 1px solid #30363d; border-radius: 4px;
      cursor: pointer; color: #c9d1d9; font-size: .82rem;
      transition: background .15s, border-color .15s;
    }}
    .radio-group input[type=radio] {{ display: none; }}
    .radio-group input[type=radio]:checked + label {{
      background: #1f6feb33; border-color: #58a6ff; color: #58a6ff;
    }}

    button#run-btn {{
      width: 100%; margin-top: 1.2rem; padding: .65rem;
      background: #238636; color: #fff; border: none;
      border-radius: 6px; font-family: inherit; font-size: .95rem;
      cursor: pointer; letter-spacing: 1px; transition: background .15s;
    }}
    button#run-btn:hover {{ background: #2ea043; }}
    button#run-btn:disabled {{ background: #21262d; color: #484f58; cursor: not-allowed; }}

    .status-pill {{
      display: inline-block; padding: .2rem .7rem; border-radius: 12px;
      font-size: .75rem; font-weight: bold; letter-spacing: 1px;
      text-transform: uppercase; margin-bottom: 1rem;
    }}
    .s-idle    {{ background: #21262d; color: #6e7681; }}
    .s-running {{ background: #1f6feb22; color: #58a6ff;
                  animation: pulse 1.5s ease-in-out infinite; }}
    .s-done    {{ background: #23863622; color: #3fb950; }}
    .s-error   {{ background: #da363322; color: #f85149; }}
    @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.5}} }}

    .progress-row {{
      display: flex; justify-content: space-between;
      font-size: .78rem; color: #8b949e; margin-bottom: .4rem;
    }}
    .bar-bg {{ background: #21262d; border-radius: 4px; height: 6px;
               overflow: hidden; margin-bottom: 1rem; }}
    .bar {{ height: 100%; background: #3fb950; border-radius: 4px;
            transition: width .4s; }}

    /* Log table */
    .log-wrap {{
      max-height: 260px; overflow-y: auto;
      border: 1px solid #21262d; border-radius: 4px;
      scrollbar-width: thin; scrollbar-color: #30363d #161b22;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: .78rem; }}
    th {{ position: sticky; top: 0; background: #1c2128;
          color: #8b949e; font-weight: normal; text-align: left;
          padding: .3rem .6rem; border-bottom: 1px solid #30363d; }}
    td {{ padding: .25rem .6rem; border-bottom: 1px solid #21262d; }}
    tr:last-child td {{ border-bottom: none; }}
    .act  {{ color: #79c0ff; }}
    .pos  {{ color: #3fb950; }}
    .zero {{ color: #6e7681; }}
    .neg  {{ color: #f85149; }}

    /* Episode summary */
    .ep-grid {{ display: flex; flex-direction: column; gap: .4rem; margin-top: .8rem; }}
    .ep-row {{
      display: flex; justify-content: space-between; align-items: center;
      background: #0d1117; border: 1px solid #21262d; border-radius: 4px;
      padding: .4rem .8rem; font-size: .82rem;
    }}
    .ep-badge {{ padding: .15rem .5rem; border-radius: 10px; font-size: .72rem; }}
    .ep-ok  {{ background: #23863622; color: #3fb950; }}
    .ep-ko  {{ background: #21262d;   color: #6e7681; }}

    /* Gallery */
    .gif-gallery {{ display: flex; flex-wrap: wrap; gap: .8rem; margin-top: .8rem; }}
    .gif-gallery figure {{
      border: 1px solid #30363d; border-radius: 6px;
      overflow: hidden; background: #0d1117;
      display: flex; flex-direction: column;
    }}
    .gif-gallery img {{ width: 160px; height: 160px; object-fit: contain;
                        image-rendering: pixelated; }}
    .gif-gallery figcaption {{
      font-size: .72rem; color: #6e7681; text-align: center; padding: .3rem;
    }}

    #vae-img {{ max-width: 100%; border-radius: 4px;
                border: 1px solid #30363d; margin-top: .6rem; }}

    .error-box {{
      background: #da363311; border: 1px solid #f85149;
      border-radius: 6px; padding: .8rem 1rem;
      color: #f85149; font-size: .82rem; margin-top: .8rem;
    }}

    header {{ text-align: center; margin-bottom: 1.5rem; }}
  </style>
</head>
<body>
<header>
  <h1>&#9889; V-M-C Inference</h1>
  <p>World Model FPGA &mdash; Prototype Grid Routing</p>
</header>

<div class="layout">

  <!-- ── Config panel ── -->
  <div>
    <div class="card">
      <div class="card-title">Configuration</div>

      <label for="episodes">Épisodes</label>
      <input type="number" id="episodes" value="5" min="1" max="50">

      <label>Politique</label>
      <div class="radio-group">
        <input type="radio" id="m-greedy" name="mode" value="greedy" checked>
        <label for="m-greedy">Greedy</label>
        <input type="radio" id="m-stoch" name="mode" value="stochastic">
        <label for="m-stoch">Stochastique</label>
      </div>

      <label for="env-sel">Environnement</label>
      <select id="env-sel">{_ENV_OPTIONS}</select>

      <button id="run-btn" onclick="startRun()">&#9654;  Lancer l'inférence</button>
    </div>

    <div class="card" style="margin-top:1.2rem;">
      <div class="card-title">Épisodes</div>
      <div id="ep-list"><span style="color:#484f58;font-size:.8rem;">—</span></div>
    </div>
  </div>

  <!-- ── Main panel ── -->
  <div style="display:flex;flex-direction:column;gap:1.2rem;">

    <!-- Progress -->
    <div class="card">
      <div class="card-title">Progression</div>
      <div id="pill" class="status-pill s-idle">Idle</div>
      <div class="progress-row">
        <span id="prog-label">—</span>
        <span id="prog-pct">—</span>
      </div>
      <div class="bar-bg"><div class="bar" id="bar" style="width:0%"></div></div>
      <div id="error-box" style="display:none" class="error-box"></div>
    </div>

    <!-- Live log -->
    <div class="card">
      <div class="card-title">Log temps réel</div>
      <div class="log-wrap">
        <table>
          <thead>
            <tr>
              <th>Ep</th><th>t</th><th>Action</th>
              <th>r</th><th>R</th><th>r&#770;</th><th>|z|</th>
            </tr>
          </thead>
          <tbody id="log-body">
            <tr><td colspan="7" style="color:#484f58;text-align:center;">—</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- GIF gallery -->
    <div class="card">
      <div class="card-title">Épisodes animés</div>
      <div id="gif-gallery" class="gif-gallery">
        <span style="color:#484f58;font-size:.8rem;">Les GIFs apparaîtront ici après chaque épisode.</span>
      </div>
    </div>

    <!-- VAE reconstruction -->
    <div class="card">
      <div class="card-title">Reconstruction VAE</div>
      <img id="vae-img" style="display:none" alt="VAE reconstruction">
    </div>

  </div>
</div>

<script>
  let _prevLogLen = 0;
  let _prevMediaLen = 0;
  let _prevEpLen = 0;
  let _polling = null;
  let _lastStatus = "idle";

  function rclass(v) {{
    if (v > 0) return "pos";
    if (v < 0) return "neg";
    return "zero";
  }}

  function startRun() {{
    const episodes = parseInt(document.getElementById("episodes").value);
    const greedy = document.querySelector('input[name=mode]:checked').value === "greedy";
    const env_id = document.getElementById("env-sel").value;

    document.getElementById("run-btn").disabled = true;
    document.getElementById("log-body").innerHTML =
      '<tr><td colspan="7" style="color:#484f58;text-align:center;">—</td></tr>';
    document.getElementById("gif-gallery").innerHTML =
      '<span style="color:#484f58;font-size:.8rem;">En attente...</span>';
    document.getElementById("ep-list").innerHTML =
      '<span style="color:#484f58;font-size:.8rem;">—</span>';
    document.getElementById("vae-img").style.display = "none";
    document.getElementById("error-box").style.display = "none";
    _prevLogLen = 0; _prevMediaLen = 0; _prevEpLen = 0;

    fetch("/inference/run", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify({{episodes, greedy, env_id}})
    }}).then(() => {{
      if (_polling) clearInterval(_polling);
      _polling = setInterval(poll, 500);
    }}).catch(err => alert("Erreur : " + err));
  }}

  async function poll() {{
    let d;
    try {{ d = await fetch("/inference/status").then(r => r.json()); }}
    catch (_) {{ return; }}

    const s = d.status;

    // Pill
    const pill = document.getElementById("pill");
    pill.className = "status-pill s-" + s;
    pill.textContent = s.charAt(0).toUpperCase() + s.slice(1);

    // Progress bar
    const total = d.progress.total || 1;
    const ep = d.progress.ep || 0;
    const step = d.progress.step || 0;
    const pct = Math.round((ep + (step > 0 ? 0.5 : 0)) / total * 100);
    document.getElementById("prog-label").textContent =
      `Épisode ${{ep + 1}} / ${{total}} — step ${{step}}`;
    document.getElementById("prog-pct").textContent = pct + "%";
    document.getElementById("bar").style.width = pct + "%";

    // Error
    if (s === "error" && d.error) {{
      const eb = document.getElementById("error-box");
      eb.style.display = "block";
      eb.textContent = d.error;
    }}

    // Log (append only)
    const log = d.log || [];
    if (log.length > _prevLogLen) {{
      const tbody = document.getElementById("log-body");
      if (_prevLogLen === 0) tbody.innerHTML = "";
      for (let i = _prevLogLen; i < log.length; i++) {{
        const e = log[i];
        const r = (v) => `<td class="${{rclass(v)}}">${{v >= 0 ? "+" : ""}}${{v}}</td>`;
        tbody.innerHTML += `<tr>
          <td>${{e.ep}}</td><td>${{e.step}}</td>
          <td class="act">${{e.action_name}}</td>
          ${{r(e.reward)}}${{r(e.total_reward)}}${{r(e.pred_reward)}}
          <td>${{e.z_norm}}</td>
        </tr>`;
      }}
      _prevLogLen = log.length;
      const wrap = document.querySelector(".log-wrap");
      wrap.scrollTop = wrap.scrollHeight;
    }}

    // Episode summary (append only)
    const eps = d.episodes || [];
    if (eps.length > _prevEpLen) {{
      const list = document.getElementById("ep-list");
      if (_prevEpLen === 0) list.innerHTML = "";
      for (let i = _prevEpLen; i < eps.length; i++) {{
        const e = eps[i];
        const badge = e.success
          ? `<span class="ep-badge ep-ok">SUCCESS</span>`
          : `<span class="ep-badge ep-ko">fail</span>`;
        list.innerHTML += `<div class="ep-row">
          <span>Ép. ${{e.ep.toString().padStart(2,"0")}}</span>
          <span style="color:#6e7681">${{e.steps}} steps</span>
          <span class="${{rclass(e.total_reward)}}">${{e.total_reward >= 0 ? "+" : ""}}${{e.total_reward.toFixed(3)}}</span>
          ${{badge}}
        </div>`;
      }}
      _prevEpLen = eps.length;
    }}

    // Media (GIFs + VAE, append only)
    const media = d.media || [];
    if (media.length > _prevMediaLen) {{
      const gallery = document.getElementById("gif-gallery");
      if (_prevMediaLen === 0) gallery.innerHTML = "";
      for (let i = _prevMediaLen; i < media.length; i++) {{
        const f = media[i];
        if (f.endsWith(".gif")) {{
          gallery.innerHTML += `<figure>
            <img src="/inference/media/${{f}}?t=${{Date.now()}}" alt="${{f}}">
            <figcaption>${{f}}</figcaption>
          </figure>`;
        }} else if (f === "vae_reconstruction.png") {{
          const img = document.getElementById("vae-img");
          img.src = "/inference/media/vae_reconstruction.png?t=" + Date.now();
          img.style.display = "block";
        }}
      }}
      _prevMediaLen = media.length;
    }}

    // Stop polling when done/error
    if ((s === "done" || s === "error") && _lastStatus !== s) {{
      clearInterval(_polling);
      document.getElementById("run-btn").disabled = false;
      if (s === "done") {{
        document.getElementById("prog-label").textContent =
          `${{total}} épisode${{total > 1 ? "s" : ""}} terminé${{total > 1 ? "s" : ""}}`;
        document.getElementById("prog-pct").textContent = "100%";
        document.getElementById("bar").style.width = "100%";
      }}
    }}
    _lastStatus = s;
  }}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
