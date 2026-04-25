"""
Utilitaires partagés entre les scripts d'entraînement.

Centralise la normalisation des observations et les constantes d'architecture.
"""

import json
import os
import datetime
import sys
from typing import Any

import numpy as np
import torch

# Constantes d'architecture partagées entre tous les scripts
LATENT_DIM: int = 256
HIDDEN_DIM: int = 1024
ACTION_DIM: int = 7
NUM_GAUSSIANS: int = 8

# MiniGrid encode (object_id, color_id, state_id) ∈ [0, 10]
_OBS_NORM: float = 10.0


def obs_to_tensor(
    obs_json: list,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Convertit l'observation JSON renvoyée par ``/reset`` ou ``/step`` en tenseur PyTorch.

    :param obs_json: Liste 3D (7, 7, 3) d'entiers renvoyée par l'API FastAPI.
    :type obs_json: list
    :param device: Device cible.
    :type device: torch.device
    :returns: Tenseur float32 normalisé dans [0, 1], shape (1, 3, 7, 7), prêt pour le VAE.
    :rtype: torch.Tensor
    """
    arr = np.array(obs_json, dtype=np.float32) / _OBS_NORM
    arr = arr.transpose(2, 0, 1)  # (7, 7, 3) → (3, 7, 7)
    return torch.tensor(arr, device=device).unsqueeze(0)


def obs_array_to_tensor(
    obs_array: np.ndarray,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Convertit un batch d'observations numpy en tenseur PyTorch normalisé.

    :param obs_array: Observations brutes, shape (B, 7, 7, 3).
    :type obs_array: np.ndarray
    :param device: Device cible.
    :type device: torch.device
    :returns: Tenseur float32 normalisé dans [0, 1], shape (B, 3, 7, 7).
    :rtype: torch.Tensor
    """
    arr = obs_array.astype(np.float32) / _OBS_NORM
    arr = arr.transpose(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
    return torch.tensor(arr, device=device)


_STATUS_PATH = os.path.join("runs", "status.json")

_history: list[dict] = []
_current_label: str = ""


def write_status(
    step: int,
    label: str,
    current: int,
    total: int,
    metrics: dict | None = None,
    eta: str | None = None,
) -> None:
    """
    Écrit l'état courant de l'entraînement dans runs/status.json.

    Accumule un historique des métriques par epoch dans la variable globale
    ``_history``. L'historique est réinitialisé automatiquement quand ``label``
    change (nouvelle phase d'entraînement). Utilise une écriture atomique
    (tmp + rename) pour éviter que le dashboard lise un fichier partiel.

    :param step: Numéro de l'étape courante (1, 2 ou 3).
    :type step: int
    :param label: Nom de la phase (ex: "World Model — VAE").
    :type label: str
    :param current: Itération courante (epoch, épisode, transitions...).
    :type current: int
    :param total: Nombre total d'itérations pour cette phase.
    :type total: int
    :param metrics: Métriques à afficher {nom: valeur}.
    :type metrics: dict | None
    :param eta: Temps restant estimé, format "HH:MM:SS".
    :type eta: str | None
    :returns: None
    """
    global _history, _current_label

    if label != _current_label:
        _history = []
        _current_label = label

    point: dict = {"x": current}
    point.update(metrics or {})
    _history.append(point)

    os.makedirs("runs", exist_ok=True)
    payload = {
        "step": step,
        "label": label,
        "current": current,
        "total": total,
        "metrics": metrics or {},
        "eta": eta or "",
        "updated": datetime.datetime.now().strftime("%H:%M:%S"),
        "history": _history,
    }
    tmp = _STATUS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, _STATUS_PATH)
