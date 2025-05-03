import torch
import gymnasium as gym
import highway_env
import numpy as np
from sac_agent import SACAgent, ReplayBuffer
from tqdm import trange
import matplotlib.pyplot as plt

ENV_NAME = "parking-v0"
NUM_EPISODES = 500
MAX_STEPS = 500
BATCH_SIZE = 256
WARMUP_STEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration de l'environnement avec actions continues
config = {
    "observation": {"type": "Kinematics"},
    "action": {"type": "ContinuousAction"},
    "duration": 50,
    "vehicles_count": 1,
    "controlled_vehicles": 1
}
env = gym.make(ENV_NAME, config=config)

# Vérification de l'espace d'action
assert isinstance(env.action_space, gym.spaces.Box), "SAC nécessite un espace d'action continu"

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = SACAgent(state_dim, action_dim, device)
buffer = ReplayBuffer()

episode_rewards = []

for episode in trange(NUM_EPISODES, desc="Training"):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)

        if len(buffer) < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = agent.policy.sample(state)
                action = action.cpu().numpy()[0]

        obs_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, obs_next, done)
        obs = obs_next
        total_reward += reward

        if len(buffer) > BATCH_SIZE:
            agent.update(buffer, BATCH_SIZE)

        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Épisode {episode + 1} — Récompense : {total_reward:.2f}")

env.close()

plt.plot(episode_rewards)
plt.title("SAC sur parking-v0")
plt.xlabel("Épisode")
plt.ylabel("Reward total")
plt.grid()
plt.show()

torch.save(agent.policy.state_dict(), "sac_policy_parking.pth")
print("✅ Modèle SAC entraîné et sauvegardé.")
