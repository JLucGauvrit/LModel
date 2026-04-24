import torch
import torch.nn as nn


class Controller(nn.Module):
    """
    Modèle C (Contrôleur) — Réseau linéaire pour la prise de décision.

    Intentionnellement minimal (une seule couche linéaire) pour être
    efficacement optimisable par REINFORCE sans sur-paramétrage.

    Prend en entrée la concaténation de z_t (Vision) et h_t (Mémoire)
    et produit des logits sur les actions discrètes de MiniGrid.

    :param latent_dim: Dimension du vecteur latent z (sortie VAE).
    :type latent_dim: int
    :param hidden_dim: Dimension de l'état caché h (sortie MDN-RNN).
    :type hidden_dim: int
    :param num_actions: Nombre d'actions disponibles (7 dans MiniGrid-Empty).
    :type num_actions: int
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        num_actions: int = 7,
    ):
        super().__init__()
        self.fc = nn.Linear(latent_dim + hidden_dim, num_actions)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Calcule les logits d'action à partir de la vision et de la mémoire.

        :param z: Vecteur latent VAE courant, shape (B, latent_dim).
        :type z: torch.Tensor
        :param h: État caché MDN-RNN courant, shape (B, hidden_dim).
        :type h: torch.Tensor
        :returns: Logits non normalisés, shape (B, num_actions).
                  À passer dans ``torch.distributions.Categorical(logits=...)``
                  pour un échantillonnage stochastique différentiable.
        :rtype: torch.Tensor
        """
        return self.fc(torch.cat([z, h], dim=-1))
