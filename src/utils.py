"""
Utilitaires partagés entre les scripts d'entraînement.

Centralise la normalisation des observations et les constantes d'architecture.
"""

import sys
from typing import Any

import numpy as np
import torch

# Constantes d'architecture partagées entre tous les scripts
LATENT_DIM: int = 32
HIDDEN_DIM: int = 256
ACTION_DIM: int = 7
NUM_GAUSSIANS: int = 5

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
