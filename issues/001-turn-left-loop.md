# Issue #001 — Agent bloqué en boucle turn_left

**Statut :** Ouvert  
**Composant :** Controller (étape 3) + collecte de données (étape 1)  
**Priorité :** Critique — bloque toute évaluation réelle de l'agent

---

## Symptôme

En inférence (`inference_ui.py`), l'agent exécute uniquement `turn_left` à chaque pas de temps,
quelle que soit l'observation. Il ne progresse jamais vers le but et ne marque aucune récompense.

---

## Diagnostic

### Cause racine 1 — Récompense sparse, gradients nuls

Dans `MiniGrid-Empty-8x8-v0`, la récompense n'est attribuée qu'à l'arrivée sur la case verte.
Avec une politique aléatoire lors de la collecte, la probabilité de trouver la sortie en moins
de 500 pas est infime (≈ 1–2 % par épisode).

Conséquence dans le pipeline :

```
r_t ≈ 0 pour presque toutes les transitions
   ↓
MDN-RNN.reward_head prédit toujours ~0
   ↓
compute_dream_returns() → retours G_t ≈ 0 pour tous les rêves
   ↓
loss REINFORCE = -Σ log π(a_t) · G_t ≈ 0
   ↓
∇loss ≈ 0 → les poids du Controller ne bougent pas
```

Le Controller reste à son initialisation aléatoire après les 8 000 updates.

### Cause racine 2 — Piège du argmax sur une politique plate

Un Controller non entraîné produit des logits quasi-uniformes, par exemple :

```
[0.1401, 0.1400, 0.1400, 0.1400, 0.1399, 0.1400, 0.1400]
```

En inférence avec `greedy=True`, `torch.argmax()` choisit systématiquement l'action 0
(`turn_left`) car elle a l'avantage infinitésimal dû à l'initialisation des poids (ex: Kaiming).

Conséquence :

```
argmax → turn_left
   ↓
L'agent tourne → observation quasi identique
   ↓
z_t ≈ z_{t-1} → même logit → turn_left → boucle infinie
```

---

## Pistes de résolution

### Fix 1 — Reward shaping (prioritaire)

Ajouter une récompense dense proportionnelle à la réduction de distance Manhattan vers le but.
À implémenter dans `1_collect_data.py` via un wrapper gymnasium.

```python
class DistanceRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if reward == 0:
            agent = self.env.agent_pos
            goal  = self.env.goal_pos  # accès interne MiniGrid
            reward = -0.01 * (abs(agent[0]-goal[0]) + abs(agent[1]-goal[1])) / self.env.width
        return obs, reward, term, trunc, info
```

### Fix 2 — Curriculum : commencer par Empty-5x5

Remplacer `MiniGrid-Empty-8x8-v0` par `MiniGrid-Empty-5x5-v0` pour la première phase
d'entraînement. La probabilité de trouver la sortie par hasard est ~10× plus élevée,
ce qui amorce les gradients dès les premières epochs.

### Fix 3 — Politique stochastique en inférence par défaut

Dans `inference_ui.py`, basculer la valeur par défaut de `greedy` à `False` jusqu'à ce que
le Controller soit réellement entraîné. Le mode stochastique évite le piège argmax même
sur une politique plate.

```python
class RunConfig(BaseModel):
    episodes: int = 5
    greedy: bool = False   # était True
    env_id: str = "MiniGrid-Empty-8x8-v0"
```

### Fix 4 — Exploration dirigée lors de la collecte

Remplacer la politique aléatoire pure dans `1_collect_data.py` par une exploration
epsilon-greedy biaisée vers `forward` (action 2), qui est statistiquement plus utile que
les rotations pour trouver la sortie.

---

## Ordre de résolution recommandé

1. **Fix 3** (immédiat, 1 ligne) — stopper la boucle en inférence
2. **Fix 2** (court terme) — amorcer l'apprentissage sur grille 5x5
3. **Fix 1** (moyen terme) — reward shaping pour signal d'apprentissage dense
4. **Fix 4** (optionnel) — améliore la qualité des données collectées
