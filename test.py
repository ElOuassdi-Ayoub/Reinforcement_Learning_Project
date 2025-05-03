import time
import torch
import numpy as np
import gymnasium as gym
import highway_env
from sac_agent import SACAgent

EVAL_EPISODES = 10
MAX_STEPS     = 500
RENDER_DELAY  = 0.02

# 1) Créer l’environnement avec ContinuousAction
env = gym.make(
    "highway-v0",
    render_mode="human",
    config={
        "observation": {"type": "Kinematics"},
        "action":      {"type": "ContinuousAction"},
        "duration":           50,
        "controlled_vehicles": 1,
        "vehicles_count":      5,
    }
)

# 2) Charger l’agent et sa politique
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = SACAgent()
agent.policy.load_state_dict(torch.load("sac_highway_pi.pth", map_location=device))
agent.policy.to(device).eval()

# 3) Boucle d’évaluation
for ep in range(1, EVAL_EPISODES + 1):
    obs, _ = env.reset()
    obs = np.asarray(obs).reshape(-1)
    tot_r = 0.0

    for _ in range(MAX_STEPS):
        env.render()
        time.sleep(RENDER_DELAY)

        s = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            a_tensor, _ = agent.policy.sample(s)
        action = a_tensor.cpu().numpy().flatten()  # vecteur de taille 2

        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        obs = np.asarray(next_obs).reshape(-1)
        tot_r += reward
        if done:
            break

    print(f"Épisode {ep}  reward={tot_r:.2f}")

env.close()
