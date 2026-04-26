"""
Dashboard de supervision d'entraînement V-M-C.

Lit runs/status.json mis à jour par les scripts d'entraînement et diffuse
les mises à jour en temps réel via Server-Sent Events (SSE).

Usage :
  python src/server.py
"""

import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

app = FastAPI()
_STATUS_FILE = Path("runs/status.json")

_UI_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>V-M-C Training</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
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
      padding: 1.5rem 2rem; width: 520px;
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
    .eta    { font-size: 0.78rem; color: #6e7681; margin-top: 0.5rem; text-align: right; }
    .metrics { display: flex; gap: 2rem; margin-top: 1.2rem; }
    .metric { font-size: 0.78rem; color: #6e7681; }
    .metric span { display: block; color: #e6edf3; font-size: 1rem; margin-bottom: 2px; }
    .updated { font-size: 0.72rem; color: #484f58; margin-top: 1rem; text-align: right; }
    .waiting { color: #6e7681; font-size: 0.9rem; text-align: center; padding: 0.5rem 0; }
    #charts { display: flex; flex-wrap: wrap; gap: 1.5rem; justify-content: center; }
    .chart-card {
      background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: 1.2rem 1.5rem; width: 520px;
    }
    .chart-title { color: #8b949e; font-size: 0.82rem; margin-bottom: 0.8rem; }
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

  <section id="charts"></section>

  <div class="footer">
    Courbes détaillées &rarr;
    <a href="/tensorboard/" target="_blank">TensorBoard</a>
  </div>

  <script>
    const CHART_COLOR = '#58a6ff';
    const CHART_BG    = 'rgba(88,166,255,0.12)';
    let currentLabel = null;
    let charts = {};

    function makeChart(metricKey) {
      const section = document.getElementById('charts');
      const card = document.createElement('div');
      card.className = 'chart-card';
      const title = document.createElement('div');
      title.className = 'chart-title';
      title.textContent = metricKey;
      const canvas = document.createElement('canvas');
      canvas.height = 140;
      card.appendChild(title);
      card.appendChild(canvas);
      section.appendChild(card);
      return new Chart(canvas, {
        type: 'line',
        data: { datasets: [{ data: [], borderColor: CHART_COLOR, backgroundColor: CHART_BG,
                              borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.3 }] },
        options: {
          animation: false,
          parsing: { xAxisKey: 'x', yAxisKey: 'y' },
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#6e7681' }, grid: { color: '#21262d' } },
            y: { ticks: { color: '#6e7681' }, grid: { color: '#21262d' } },
          }
        }
      });
    }

    function resetCharts() { document.getElementById('charts').innerHTML = ''; charts = {}; }

    function updateCharts(history) {
      if (!history || !history.length) return;
      for (const key of Object.keys(history[0]).filter(k => k !== 'x')) {
        if (!charts[key]) charts[key] = makeChart(key);
        charts[key].data.datasets[0].data = history.map(p => ({ x: p.x, y: p[key] }));
        charts[key].update('none');
      }
    }

    function render(d) {
      if (!d.label || d.step === 0) return;
      if (d.label !== currentLabel) { currentLabel = d.label; resetCharts(); }
      const pct = d.total > 0 ? Math.min(100, Math.round(d.current / d.total * 100)) : 0;
      let dots = '';
      for (let i = 1; i <= 3; i++) {
        const cls = i < d.step ? 'done' : i === d.step ? 'active' : '';
        dots += `<div class="step-dot ${cls}" title="Etape ${i}"></div>`;
      }
      let metrics = '';
      for (const [k, v] of Object.entries(d.metrics || {})) {
        const fmt = typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(4) : v;
        metrics += `<div class="metric"><span>${fmt}</span>${k}</div>`;
      }
      document.getElementById('card').innerHTML = `
        <div class="card-header"><div class="label">${d.label}</div><div class="steps">${dots}</div></div>
        <div class="progress-row"><span>Iteration ${d.current} / ${d.total}</span><span>${pct}%</span></div>
        <div class="bar-bg"><div class="bar" style="width:${pct}%"></div></div>
        ${d.eta ? `<div class="eta">ETA ${d.eta}</div>` : ''}
        ${metrics ? `<div class="metrics">${metrics}</div>` : ''}
        <div class="updated">mis a jour ${d.updated}</div>`;
      updateCharts(d.history || []);
    }

    const es = new EventSource('/dashboard/events');
    es.onmessage = (e) => { try { render(JSON.parse(e.data)); } catch (_) {} };
  </script>
</body>
</html>"""


async def _event_stream():
    """
    Generateur asynchrone qui surveille runs/status.json et emet un event SSE
    a chaque modification du fichier (detection par mtime, polling 200 ms).

    :returns: Chaines SSE formatees ``data: <json>\\n\\n``.
    :rtype: AsyncGenerator[str, None]
    """
    last_mtime = 0.0
    while True:
        try:
            mtime = _STATUS_FILE.stat().st_mtime if _STATUS_FILE.exists() else 0.0
            if mtime != last_mtime:
                last_mtime = mtime
                yield f"data: {_STATUS_FILE.read_text()}\n\n"
        except Exception:
            pass
        await asyncio.sleep(0.2)


@app.get("/events")
async def events() -> StreamingResponse:
    """
    Endpoint Server-Sent Events : diffuse chaque mise a jour de status.json
    des qu'elle est ecrite par les scripts d'entrainement.

    :returns: Stream SSE en ``text/event-stream``.
    :rtype: StreamingResponse
    """
    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/status")
def status() -> JSONResponse:
    """
    Retourne l'etat courant de l'entrainement depuis runs/status.json.

    :returns: JSON ``{step, label, current, total, metrics, eta, updated, history}``
              ou objet vide si l'entrainement n'a pas encore demarre.
    :rtype: JSONResponse
    """
    if not _STATUS_FILE.exists():
        return JSONResponse({"step": 0, "label": "", "current": 0, "total": 0,
                             "metrics": {}, "eta": "", "updated": "", "history": []})
    return JSONResponse(json.loads(_STATUS_FILE.read_text()))


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    """
    Sert le dashboard HTML de supervision d'entrainement.

    :returns: Page HTML complete.
    :rtype: str
    """
    return _UI_HTML


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
