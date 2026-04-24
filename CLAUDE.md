# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**World Model FPGA - Prototype Grid Routing**: A prototype for applying World Models to FPGA Electronic Design Automation. The current goal is training an agent to understand the dynamics of a routing grid (via a MiniGrid abstraction) using a distributed V-M-C (Vision-Memory-Controller) architecture.

## Running the Project

**Preferred method — Docker Compose (both services together):**
```bash
docker-compose up
```

**Direct execution (without Docker):**
```bash
# Terminal 1
python src/serveur.py

# Terminal 2 (server URL hardcoded to http://env_server:8000 — change it for local runs)
python src/client.py
```

> `client.py` hardcodes `SERVER_URL = "http://env_server:8000"` (the Docker service name). For local runs outside Docker, this must be changed to `http://localhost:8000`.

## Architecture

Two decoupled services in separate containers:

| Service | File | Role |
|---|---|---|
| `env_server` | `src/serveur.py` | FastAPI + MiniGrid simulation, exposes `GET /reset` and `POST /step` |
| `trainer` | `src/client.py` | Training loop — sends actions to the server, receives observations |

**API contract (`env_server`):**
- `GET /reset` → `{ "obs": [[[int]]] }` (MiniGrid image as nested list)
- `POST /step` with `{ "action": int }` → `{ "obs": ..., "reward": float, "done": bool }`

## Tech Stack

- **Python 3.10** (pinned in Dockerfile)
- **gymnasium + minigrid** — RL environment (`MiniGrid-Empty-8x8-v0`)
- **FastAPI + uvicorn** — simulation server
- **requests** — HTTP client in trainer
- **PyTorch** — intended for model definitions (not yet implemented)

## Current State & Planned Work

This is an early prototype (hardcoded action=2, no learning yet). The README mentions a `src/models/` directory for VAE, LSTM, and Controller definitions — this does not exist yet and is the next implementation step.
