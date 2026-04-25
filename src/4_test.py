import os
import torch
import numpy as np
import gymnasium as gym
import minigrid
from PIL import Image

from models import VAE, MDNRNN, Controller
from utils import obs_array_to_tensor, LATENT_DIM, HIDDEN_DIM, ACTION_DIM, NUM_GAUSSIANS

CHECKPOINT_DIR = "checkpoints"

def load_all_models(device):
    vae = VAE(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "vae.pt"), map_location=device))
    
    mdn_rnn = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, num_gaussians=NUM_GAUSSIANS).to(device)
    mdn_rnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "mdn_rnn.pt"), map_location=device))
    
    controller = Controller(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, num_actions=ACTION_DIM).to(device)
    controller.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "controller.pt"), map_location=device))
    
    vae.eval()
    mdn_rnn.eval()
    controller.eval()
    return vae, mdn_rnn, controller

def test_agent(episodes=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae, mdn_rnn, controller = load_all_models(device)
    
    # On demande explicitement le rendu rgb_array pour générer le GIF
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    
    os.makedirs("data", exist_ok=True)

    for ep in range(episodes):
        gym_obs, _ = env.reset()
        obs = obs_array_to_tensor(gym_obs["image"][np.newaxis], device)
        h, c = mdn_rnn.init_hidden(1, device)
        
        frames = []
        done = False
        step = 0
        total_reward = 0.0

        while not done:
            frames.append(Image.fromarray(env.render()))
            
            with torch.no_grad():
                z, _ = vae.encode(obs)
                action_logits = controller(z, h)
                
                # En test, on prend l'action avec la plus forte probabilité (argmax)
                # au lieu de faire un sample stochastique
                action = torch.argmax(action_logits, dim=-1)
                
                gym_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                total_reward += reward
                
                if not done:
                    obs = obs_array_to_tensor(gym_obs["image"][np.newaxis], device)
                    _, _, _, h, c = mdn_rnn(z, action, h, c)
            
            step += 1

        # Sauvegarde du dernier frame
        frames.append(Image.fromarray(env.render()))
        
        gif_path = f"data/test_ep_{ep}.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        
        print(f"Épisode {ep} | Steps: {step} | Reward: {total_reward:.3f} | GIF sauvegardé: {gif_path}")

    env.close()

if __name__ == "__main__":
    test_agent()
    