"""
Dashboard de supervision d'entraînement V-M-C.

Lit runs/status.json mis à jour par les scripts d'entraînement
et sert une UI web rafraîchie toutes les secondes.

Usage :
  python src/server.py
"""

import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()
_STATUS_FILE = Path("runs/status.json")

_UI_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>V-M-C Training</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d1117; color: #c9d1d9;
      font-family: 'Courier New', Courier, monospace;
      display: flex; flex-direction: column; align-items: center;
      min-height: 100vh; padding: 2rem 1rem; gap: 2rem;
    }
    header { text-align: center; }
    header h1 { color: #58a6ff; font-size: 1.4rem; letter-spacing: 3px; text-transform: uppercase; }
    header p  { color: #6e7681; font-size: 0.8rem; margin-top: 0.3rem; }
    .card {
      background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: 1.5rem 2rem; width: 480px;
    }
    .card-header {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 1.2rem;
    }
    .label { color: #58a6ff; font-size: 1.05rem; }
    .steps { display: flex; gap: 6px; }
    .step-dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: #21262d; border: 1px solid #30363d;
    }
    .step-dot.done   { background: #3fb950; border-color: #3fb950; }
    .step-dot.active { background: #58a6ff; border-color: #58a6ff;
                       animation: pulse 1.5s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
    .progress-row {
      display: flex; justify-content: space-between;
      font-size: 0.82rem; color: #8b949e; margin-bottom: 6px;
    }
    .bar-bg { background: #21262d; border-radius: 4px; height: 8px; overflow: hidden; }
    .bar    { height: 100%; background: #3fb950; border-radius: 4px; transition: width 0.5s; }
    .metrics { display: flex; gap: 2rem; margin-top: 1.2rem; }
    .metric { font-size: 0.78rem; color: #6e7681; }
    .metric span { display: block; color: #e6edf3; font-size: 1rem; margin-bottom: 2px; }
    .updated { font-size: 0.72rem; color: #484f58; margin-top: 1rem; text-align: right; }
    .waiting { color: #6e7681; font-size: 0.9rem; text-align: center; padding: 0.5rem 0; }
    .footer { font-size: 0.78rem; color: #6e7681; }
    .footer a { color: #58a6ff; text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <header>
    <h1>&#9889; V-M-C Training</h1>
    <p>World Model FPGA &mdash; Prototype Grid Routing</p>
  </header>

  <div class="card" id="card">
    <div class="waiting">En attente du démarrage...</div>
  </div>

  <div class="footer">
    Courbes détaillées &rarr;
    <a href="http://localhost:6006" target="_blank">TensorBoard :6006</a>
  </div>

  <script>
    async function refresh() {
      try {
        const d = await fetch('/status').then(r => r.json());
        if (!d.label || d.step === 0) return;

        const pct = d.total > 0 ? Math.min(100, Math.round(d.current / d.total * 100)) : 0;

        let dots = '';
        for (let i = 1; i <= 3; i++) {
          const cls = i < d.step ? 'done' : i === d.step ? 'active' : '';
          dots += `<div class="step-dot ${cls}" title="Étape ${i}"></div>`;
        }

        let metrics = '';
        for (const [k, v] of Object.entries(d.metrics || {})) {
          const fmt = typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(4) : v;
          metrics += `<div class="metric"><span>${fmt}</span>${k}</div>`;
        }

        document.getElementById('card').innerHTML = `
          <div class="card-header">
            <div class="label">${d.label}</div>
            <div class="steps">${dots}</div>
          </div>
          <div class="progress-row">
            <span>Itération ${d.current} / ${d.total}</span>
            <span>${pct}%</span>
          </div>
          <div class="bar-bg"><div class="bar" style="width:${pct}%"></div></div>
          ${metrics ? `<div class="metrics">${metrics}</div>` : ''}
          <div class="updated">mis à jour ${d.updated}</div>
        `;
      } catch (_) {}
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>"""


@app.get("/status")
def status() -> JSONResponse:
    """
    Retourne l'état courant de l'entraînement depuis runs/status.json.

    :returns: JSON ``{step, label, current, total, metrics, updated}``
              ou objet vide si l'entraînement n'a pas encore démarré.
    :rtype: JSONResponse
    """
    if not _STATUS_FILE.exists():
        return JSONResponse({"step": 0, "label": "", "current": 0, "total": 0,
                             "metrics": {}, "updated": ""})
    return JSONResponse(json.loads(_STATUS_FILE.read_text()))


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    """
    Sert le dashboard HTML de supervision d'entraînement.

    :returns: Page HTML complète.
    :rtype: str
    """
    return _UI_HTML


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
