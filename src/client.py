import requests

# Dans Docker, le nom du service sert d'URL
SERVER_URL = "http://env_server:8000"

def train():
    # 1. Reset l'environnement
    res = requests.get(f"{SERVER_URL}/reset").json()
    print("Observation reçue.")

    # 2. Boucle d'entraînement
    for _ in range(10):
        # L'agent choisit une action (ici arbitraire)
        action_payload = {"action": 2} 
        
        # Envoi de l'action au serveur
        res = requests.post(f"{SERVER_URL}/step", json=action_payload).json()
        
        print(f"Récompense: {res['reward']} | Fini: {res['done']}")
        
        if res['done']:
            break

if __name__ == "__main__":
    train()
    