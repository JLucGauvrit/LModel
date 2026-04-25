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

# Ajoute src/ au path pour que les imports inter-scripts fonctionnent
sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR       = "data"
CHECKPOINT_DIR = "checkpoints"


def _banner(title: str) -> None:
    """Affiche un séparateur visuel dans les logs."""
    sep = "=" * 60
    print(f"\n{sep}\n  {title}\n{sep}", flush=True)


def _has_data() -> bool:
    """
    Vérifie si des données collectées sont déjà présentes.

    :returns: ``True`` si au moins un fichier .npz existe dans ``data/``.
    :rtype: bool
    """
    return bool(glob.glob(os.path.join(DATA_DIR, "episode_*.npz")))


def _has_world_model() -> bool:
    """
    Vérifie si les checkpoints du World Model sont présents.

    :returns: ``True`` si vae.pt et mdn_rnn.pt existent dans ``checkpoints/``.
    :rtype: bool
    """
    return all(
        os.path.exists(os.path.join(CHECKPOINT_DIR, name))
        for name in ("vae.pt", "mdn_rnn.pt")
    )


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
        _banner("ÉTAPE 1 — Collecte de données (10 000 transitions)")
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
