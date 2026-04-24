import torch
import torch.nn as nn


class VisionModel(nn.Module):
    """
    Modèle V (Vision) — Encodeur convolutif de l'observation MiniGrid.

    Compresse une image 3×7×7 (channels, hauteur, largeur) en un vecteur latent
    de dimension ``latent_dim``. Remplace le VAE complet pour ce prototype.

    :param latent_dim: Dimension du vecteur latent produit en sortie.
    :type latent_dim: int
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode une observation en vecteur latent.

        :param x: Image de l'environnement, shape ``(batch, 3, 7, 7)``, valeurs float32.
        :type x: torch.Tensor
        :returns: Vecteur latent de shape ``(batch, latent_dim)``.
        :rtype: torch.Tensor
        """
        return self.conv(x)


class MemoryModel(nn.Module):
    """
    Modèle M (Mémoire/Dynamique) — Cellule LSTM qui maintient l'état caché de l'agent.

    Prend en entrée le vecteur latent courant et l'action exécutée, et prédit
    le nouvel état caché ``(h, c)`` représentant la dynamique du monde.

    :param latent_dim: Dimension du vecteur latent issu du VisionModel.
    :type latent_dim: int
    :param action_dim: Dimension de l'action (1 pour une action scalaire).
    :type action_dim: int
    :param hidden_dim: Dimension de l'état caché du LSTM.
    :type hidden_dim: int
    """

    def __init__(self, latent_dim: int = 32, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.rnn = nn.LSTMCell(latent_dim + action_dim, hidden_dim)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        h_x: torch.Tensor,
        c_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Met à jour l'état caché LSTM à partir de l'observation latente et de l'action.

        :param z: Vecteur latent courant, shape ``(batch, latent_dim)``.
        :type z: torch.Tensor
        :param action: Action scalaire choisie par le Contrôleur, shape ``(batch,)``.
        :type action: torch.Tensor
        :param h_x: État caché précédent du LSTM, shape ``(batch, hidden_dim)``.
        :type h_x: torch.Tensor
        :param c_x: État cellule précédent du LSTM, shape ``(batch, hidden_dim)``.
        :type c_x: torch.Tensor
        :returns: Tuple ``(h_new, c_new)`` — nouvel état caché et cellule du LSTM.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        action_tensor = action.unsqueeze(1).float()
        rnn_input = torch.cat([z, action_tensor], dim=1)
        return self.rnn(rnn_input, (h_x, c_x))


class Controller(nn.Module):
    """
    Modèle C (Contrôleur) — Réseau de décision qui choisit l'action à exécuter.

    Combine le vecteur latent de la Vision et l'état caché de la Mémoire pour
    produire des logits sur les 7 actions disponibles dans MiniGrid.

    :param latent_dim: Dimension du vecteur latent issu du VisionModel.
    :type latent_dim: int
    :param hidden_dim: Dimension de l'état caché issu du MemoryModel.
    :type hidden_dim: int
    :param num_actions: Nombre d'actions disponibles (7 dans MiniGrid-Empty).
    :type num_actions: int
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, num_actions: int = 7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Calcule les logits des actions à partir de la vision et de la mémoire.

        :param z: Vecteur latent courant, shape ``(batch, latent_dim)``.
        :type z: torch.Tensor
        :param h: État caché du LSTM, shape ``(batch, hidden_dim)``.
        :type h: torch.Tensor
        :returns: Logits non normalisés sur les actions, shape ``(batch, num_actions)``.
                  À passer dans ``torch.distributions.Categorical`` pour l'échantillonnage.
        :rtype: torch.Tensor
        """
        x = torch.cat([z, h], dim=1)
        return self.fc(x)
