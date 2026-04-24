"""
Utilitaires partagés entre les scripts d'entraînement.

Centralise la normalisation des observations, les constantes d'architecture
et la logique de retry HTTP pour la communication avec env_server.
"""

import sys
import time
from typing import Any

import numpy as np
import requests
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


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    max_retries: int = 10,
    initial_delay: float = 1.0,
    **kwargs: Any,
) -> requests.Response:
    """
    Effectue une requête HTTP avec retry exponentiel en cas d'échec de connexion.

    Utile au démarrage du conteneur ``trainer`` où ``env_server`` peut ne pas
    être encore prêt à répondre malgré le ``depends_on`` de Docker Compose.

    :param session: Session requests réutilisable (conserve les connexions TCP).
    :type session: requests.Session
    :param method: Méthode HTTP (``"GET"``, ``"POST"``).
    :type method: str
    :param url: URL complète de l'endpoint cible.
    :type url: str
    :param max_retries: Nombre maximum de tentatives avant abandon.
    :type max_retries: int
    :param initial_delay: Délai initial en secondes, doublé à chaque tentative (max 30 s).
    :type initial_delay: float
    :returns: Réponse HTTP avec statut 2xx.
    :rtype: requests.Response
    :raises SystemExit: Si le serveur est inaccessible après toutes les tentatives.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            resp = session.request(method, url, timeout=10, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt < max_retries - 1:
                print(f"[Retry {attempt + 1}/{max_retries}] {url} — {exc} — attente {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
            else:
                print(f"Erreur fatale : {url} inaccessible après {max_retries} tentatives.")
                sys.exit(1)
