import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Modèle V (Vision) — Autoencodeur Variationnel pour les observations MiniGrid.

    Compresse une image 3×7×7 en un vecteur latent gaussien de dimension ``latent_dim``
    via le trick de reparamétrisation, puis la reconstruit.

    Architecture encoder : Conv(3→32, stride 2) → Conv(32→64, stride 2) → Linear → (μ, log σ²)
    Architecture decoder : Linear → ConvTranspose(64→32) → ConvTranspose(32→3)

    :param latent_dim: Dimension du vecteur latent z.
    :type latent_dim: int
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder : (B, 3, 7, 7) → (B, latent_dim*2)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),  # → (B, 128, 4, 4)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # → (B, 256, 2, 2)
            nn.ReLU(),
            nn.Flatten(),                                             # → (B, 1024)
            nn.Linear(1024, latent_dim * 2),
        )

        # Decoder : (B, latent_dim) → (B, 3, 7, 7)
        self.decoder_fc = nn.Linear(latent_dim, 1024)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # → (B, 128, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1),   # → (B, 3, 7, 7)
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode une observation en paramètres de la distribution latente q(z|x).

        :param x: Image normalisée dans [0, 1], shape (B, 3, 7, 7).
        :type x: torch.Tensor
        :returns: Tuple (mu, log_var), chacun de shape (B, latent_dim).
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Trick de reparamétrisation : z = μ + ε·σ avec ε ~ N(0, I).

        Permet la rétropropagation à travers l'échantillonnage stochastique.
        En mode ``eval()``, retourne directement μ (mode de la distribution).

        :param mu: Moyenne de la distribution latente, shape (B, latent_dim).
        :type mu: torch.Tensor
        :param log_var: Log-variance, shape (B, latent_dim).
        :type log_var: torch.Tensor
        :returns: Vecteur latent z, shape (B, latent_dim).
        :rtype: torch.Tensor
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Décode un vecteur latent en reconstruction de l'observation.

        :param z: Vecteur latent, shape (B, latent_dim).
        :type z: torch.Tensor
        :returns: Observation reconstruite dans [0, 1], shape (B, 3, 7, 7).
        :rtype: torch.Tensor
        """
        h = F.relu(self.decoder_fc(z)).view(-1, 256, 2, 2)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passe complète encode → reparamétrisation → décode.

        :param x: Image normalisée dans [0, 1], shape (B, 3, 7, 7).
        :type x: torch.Tensor
        :returns: Tuple (x_recon, mu, log_var).
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    @staticmethod
    def loss(
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Loss ELBO = reconstruction MSE + β·KL divergence.

        :param x: Image originale, shape (B, 3, 7, 7).
        :type x: torch.Tensor
        :param x_recon: Image reconstruite, shape (B, 3, 7, 7).
        :type x_recon: torch.Tensor
        :param mu: Moyenne latente, shape (B, latent_dim).
        :type mu: torch.Tensor
        :param log_var: Log-variance latente, shape (B, latent_dim).
        :type log_var: torch.Tensor
        :param beta: Coefficient KL (β-VAE). 1.0 = VAE standard.
        :type beta: float
        :returns: Loss scalaire.
        :rtype: torch.Tensor
        """
        recon = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon + beta * kl
