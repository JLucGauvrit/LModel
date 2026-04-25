# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**World Model FPGA - Prototype Grid Routing** — prototype applying the Ha & Schmidhuber World Model (V-M-C) to FPGA routing automation. An RL agent learns to navigate a MiniGrid grid; the long-term goal is transferring this to real FPGA routing graphs.

## Architecture V-M-C

Le conteneur trainer instancie l'environnement MiniGrid localement pour des performances optimales et gère l'entraînement complet.

| Model | Class | File | Role |
|---|---|---|---|
| V (Vision) | `VAE` | `src/models/vae.py` | CNN encoder/decoder, 3×7×7 → z (dim 32) |
| M (Memory) | `MDNRNN` | `src/models/mdn_rnn.py` | LSTMCell + Mixture Density Network, predicts p(z_{t+1}\|z_t, a_t) |
| C (Controller) | `Controller` | `src/models/controller.py` | Single linear layer, [z, h] → 7 action logits, trained by REINFORCE |

**Shared constants** (all in `src/utils.py`): `LATENT_DIM=32`, `HIDDEN_DIM=256`, `ACTION_DIM=7`, `NUM_GAUSSIANS=5`.

## Docker Setup

One image for training and monitoring:

| Dockerfile | Service | Key deps |
|---|---|---|
| `Dockerfile.trainer` | `trainer` + `tensorboard` | torch (CUDA 12.4), numpy, tensorboard, pdoc, gymnasium, minigrid |

`Dockerfile.trainer` installs torch in a separate layer before `requirements.trainer.txt` to maximise Docker layer caching (torch ~2.5 GB stays cached across code changes).

## Running

```bash
docker compose up -d --build      # build + start all services
docker compose watch               # enable hot-reload (separate terminal)
```

**Training pipeline — run in order:**
```bash
docker compose exec trainer python src/train_all.py  # runs all steps sequentially
# or individually:
docker compose exec trainer python src/1_collect_data.py   # collect 10k transitions locally
docker compose exec trainer python src/2_train_world.py    # train VAE then MDN-RNN
docker compose exec trainer python src/3_train_controller.py  # REINFORCE on Controller
```

## Live Dashboards

| URL | Content |
|---|---|
| `http://localhost:6006` | TensorBoard — training curves |

## TensorBoard Metrics

| Tag | Script | Description |
|---|---|---|
| `VAE/loss_total`, `VAE/loss_reconstruction`, `VAE/loss_kl` | `2_train_world.py` | Per epoch |
| `MDNRNN/loss_nll` | `2_train_world.py` | Per epoch |
| `Controller/reward`, `Controller/reward_avg` | `3_train_controller.py` | Per episode (avg window = 20) |
| `Controller/episode_steps`, `Controller/policy_loss` | `3_train_controller.py` | Per episode |

Logs written to `runs/world_model/` and `runs/controller/`, mounted as a host volume.

## GPU

`CUDA_VISIBLE_DEVICES=0,1` (both GPUs exposed). Host has GT 1030 (device 0, compute 6.1) and RTX 2060 SUPER (device 1, compute 7.5). PyTorch will auto-select device 0 unless specified; to force the 2060 SUPER use `device = torch.device("cuda:1")` or restrict via `CUDA_VISIBLE_DEVICES=1`.

## Docker Compose Watch

| Service | Trigger | Action |
|---|---|---|
| `trainer` | `src/` | `sync` — files copied, scripts re-run manually |
| `trainer` | `requirements.trainer.txt` | `rebuild` |

## PyDoc

All public functions and classes use Sphinx-style docstrings (`:param:`, `:type:`, `:returns:`, `:rtype:`). GitHub Actions auto-deploys to GitHub Pages on push to `main` or `dev`:

- `main` → `https://<user>.github.io/<repo>/`
- `dev` → `https://<user>.github.io/<repo>/dev/`
