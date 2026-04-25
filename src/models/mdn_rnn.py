import torch
import torch.nn as nn


class MDNRNN(nn.Module):
    """
    Modèle M (Mémoire) — LSTM couplé à un Mixture Density Network.

    Maintient un état caché récurrent et prédit la distribution du prochain
    vecteur latent z_{t+1} sous forme d'un mélange de gaussiennes p(z_{t+1} | z_t, a_t, h_t).

    :param latent_dim: Dimension du vecteur latent z (sortie du VAE).
    :type latent_dim: int
    :param action_dim: Nombre d'actions (7 pour MiniGrid), encodées en one-hot.
    :type action_dim: int
    :param hidden_dim: Dimension de l'état caché LSTM.
    :type hidden_dim: int
    :param num_gaussians: Nombre de composantes dans le mélange de gaussiennes.
    :type num_gaussians: int
    """

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_gaussians: int = 5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians

        # Input : [z_t || one_hot(a_t)]
        self.lstm = nn.LSTMCell(latent_dim + action_dim, hidden_dim)

        # Projection MDN : hidden → (pi, mu, log_sigma) pour chaque gaussienne
        # Taille : num_gaussians * (1 + latent_dim + latent_dim)
        self.mdn_head = nn.Linear(hidden_dim, num_gaussians * (1 + 2 * latent_dim))

        # Prédiction scalaire de la récompense r_t
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Un pas de temps : met à jour l'état caché et prédit la distribution de z_{t+1} et r_t.

        :param z: Vecteur latent courant, shape (B, latent_dim).
        :type z: torch.Tensor
        :param action: Indice d'action scalaire, shape (B,).
        :type action: torch.Tensor
        :param h: État caché LSTM précédent, shape (B, hidden_dim).
        :type h: torch.Tensor
        :param c: État cellule LSTM précédent, shape (B, hidden_dim).
        :type c: torch.Tensor
        :returns: Tuple (pi, mu, sigma, reward, h_new, c_new).

                  - pi : logits de mélange (B, num_gaussians)
                  - mu : moyennes (B, num_gaussians, latent_dim)
                  - sigma : écarts-types > 0 (B, num_gaussians, latent_dim)
                  - reward : récompense prédite (B, 1)
                  - h_new, c_new : nouvel état caché LSTM
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        a_onehot = torch.zeros(z.size(0), self.action_dim, device=z.device)
        a_onehot.scatter_(1, action.view(-1, 1).long(), 1.0)

        h_new, c_new = self.lstm(torch.cat([z, a_onehot], dim=1), (h, c))

        out = self.mdn_head(h_new)
        K, D = self.num_gaussians, self.latent_dim

        pi        = out[:, :K]
        mu        = out[:, K : K * (1 + D)].view(-1, K, D)
        log_sigma = out[:, K * (1 + D) :].view(-1, K, D)
        sigma     = torch.exp(log_sigma).clamp(min=1e-4)

        reward = self.reward_head(h_new)  # (B, 1)

        return pi, mu, sigma, reward, h_new, c_new

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialise l'état caché LSTM à zéro pour un nouvel épisode.

        :param batch_size: Taille du batch.
        :type batch_size: int
        :param device: Device cible (CPU/GPU).
        :type device: torch.device
        :returns: Tuple (h_0, c_0), shape (batch_size, hidden_dim) chacun.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    @staticmethod
    def sample_latent(
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Échantillonne z_{t+1} depuis la distribution de mélange de gaussiennes.

        Sélectionne une composante k ~ Categorical(π), puis z ~ N(μ_k, σ_k).

        :param pi: Logits de mélange, shape (B, num_gaussians).
        :type pi: torch.Tensor
        :param mu: Moyennes, shape (B, num_gaussians, latent_dim).
        :type mu: torch.Tensor
        :param sigma: Écarts-types, shape (B, num_gaussians, latent_dim).
        :type sigma: torch.Tensor
        :returns: z_next échantillonné, shape (B, latent_dim).
        :rtype: torch.Tensor
        """
        mix_idx = torch.distributions.Categorical(logits=pi).sample()  # (B,)
        B, _, D = mu.shape
        idx = mix_idx.view(B, 1, 1).expand(B, 1, D)
        sel_mu    = mu.gather(1, idx).squeeze(1)     # (B, D)
        sel_sigma = sigma.gather(1, idx).squeeze(1)  # (B, D)
        return sel_mu + torch.randn_like(sel_sigma) * sel_sigma

    @staticmethod
    def mdn_loss(
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        z_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative log-likelihood du mélange de gaussiennes (log-sum-exp numériquement stable).

        log p(z_{t+1}) = logsumexp_k [ log π_k + log N(z_{t+1} | μ_k, σ_k) ]

        :param pi: Logits de mélange non normalisés, shape (B, num_gaussians).
        :type pi: torch.Tensor
        :param mu: Moyennes des gaussiennes, shape (B, num_gaussians, latent_dim).
        :type mu: torch.Tensor
        :param sigma: Écarts-types des gaussiennes, shape (B, num_gaussians, latent_dim).
        :type sigma: torch.Tensor
        :param z_next: Prochain vecteur latent cible, shape (B, latent_dim).
        :type z_next: torch.Tensor
        :returns: NLL moyenne sur le batch.
        :rtype: torch.Tensor
        """
        z_next = z_next.unsqueeze(1).expand_as(mu)
        log_prob = torch.distributions.Normal(mu, sigma).log_prob(z_next).sum(dim=-1)
        log_pi = torch.log_softmax(pi, dim=-1)
        return -torch.logsumexp(log_pi + log_prob, dim=-1).mean()
