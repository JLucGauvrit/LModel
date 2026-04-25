"""
Pipeline complet d'entraînement V-M-C.

Exécute les trois étapes dans l'ordre, avec détection automatique
des étapes déjà complétées (reprise possible après interruption) :

  Étape 1 — Collecte de données   (ignorée si data/*.npz existe)
  Étape 2 — World Model VAE+MDN   (ignorée si checkpoints existent)
  Étape 3 — Contrôleur REINFORCE  (toujours exécutée — reprend le training)

L'environnement MiniGrid est instancié localement pour des performances optimales.

Usage :
  python src/train_all.py
"""

import glob
import os
import runpy
import sys

import numpy as np

# Ajoute src/ au path pour que les imports inter-scripts fonctionnent
sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR            = "data"
CHECKPOINT_DIR      = "checkpoints"
MIN_TRANSITIONS     = 45_000  # 90 % de TARGET_TRANSITIONS (50 000)


def _banner(title: str) -> None:
    """Affiche un séparateur visuel dans les logs."""
    sep = "=" * 60
    print(f"\n{sep}\n  {title}\n{sep}", flush=True)


def _has_data() -> bool:
    """
    Vérifie si suffisamment de données sont présentes (≥ MIN_TRANSITIONS).

    :returns: ``True`` si le total de transitions dépasse MIN_TRANSITIONS.
    :rtype: bool
    """
    files = glob.glob(os.path.join(DATA_DIR, "episode_*.npz"))
    if not files:
        return False
    total = sum(int(np.load(f)["actions"].shape[0]) for f in files)
    if total < MIN_TRANSITIONS:
        print(f"  → Données insuffisantes ({total} < {MIN_TRANSITIONS}) — recollecte.")
        return False
    return True


def _has_world_model() -> bool:
    """
    Vérifie si les checkpoints du World Model sont présents ET à jour.

    Le MDN-RNN doit contenir ``reward_head.weight`` (ajouté pour les dream
    rollouts). Si la clé est absente, le checkpoint est considéré obsolète
    et l'étape 2 est relancée pour ajouter la prédiction de récompense.

    :returns: ``True`` si vae.pt et mdn_rnn.pt existent et sont à jour.
    :rtype: bool
    """
    paths = {name: os.path.join(CHECKPOINT_DIR, name) for name in ("vae.pt", "mdn_rnn.pt")}
    if not all(os.path.exists(p) for p in paths.values()):
        return False
    import torch
    ckpt = torch.load(paths["mdn_rnn.pt"], map_location="cpu", weights_only=True)
    if "reward_head.weight" not in ckpt:
        print("  → MDN-RNN sans reward_head détecté — réentraînement de l'étape 2.")
        return False
    return True


def _run(script_name: str) -> None:
    """
    Exécute un script Python dans le même processus via ``runpy``.

    :param script_name: Nom du fichier (ex: ``"1_collect_data.py"``).
    :type script_name: str
    """
    path = os.path.join(os.path.dirname(__file__), script_name)
    runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    # --- Étape 1 ---
    if _has_data():
        print(f"→ Données existantes dans {DATA_DIR}/ — collecte ignorée.", flush=True)
    else:
        _banner("ÉTAPE 1 — Collecte de données (50 000 transitions)")
        _run("1_collect_data.py")

    # --- Étape 2 ---
    if _has_world_model():
        print(f"→ Checkpoints trouvés dans {CHECKPOINT_DIR}/ — World Model ignoré.", flush=True)
    else:
        _banner("ÉTAPE 2 — Entraînement World Model (VAE + MDN-RNN)")
        _run("2_train_world.py")

    # --- Étape 3 ---
    _banner("ÉTAPE 3 — Entraînement Contrôleur (REINFORCE)")
    _run("3_train_controller.py")
