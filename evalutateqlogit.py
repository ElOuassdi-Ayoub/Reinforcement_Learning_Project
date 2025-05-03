import torch
import gymnasium as gym
import highway_env
import numpy as np
from dqn_agent import DQNAgent

def evaluate_pi_rec(env_id, model_path, num_episodes=5, max_steps=500):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crée l'env sans mirroring pour l'affichage
    env = gym.make(env_id, render_mode="human", config={
        "observation": {"type": "Kinematics", "vehicles_count": 5},
        "duration": 60,
        "vehicles_count": 30,
        "controlled_vehicles": 1,
        "lanes_count": 4,
    })

    # Dimensions
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.n

    # Initialise pi_rec
    pi_rec = DQNAgent(obs_dim, act_dim, device)
    pi_rec.q_network.load_state_dict(torch.load(model_path, map_location=device))
    # Pour évaluation, on désactive l'epsilon-greedy
    pi_rec.epsilon = 0.0

    for ep in range(1, num_episodes+1):
        obs, _ = env.reset()
        state = obs.flatten()
        done = False
        total_reward = 0.0

        for t in range(max_steps):
            # Sélection de l'action par pi_rec (sans exploration)
            action = pi_rec.select_action(state)

            # Execute et render
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            next_state = obs.flatten()

            total_reward += reward
            state = next_state

            if done or truncated:
                break

        print(f"Episode {ep}: total_reward={total_reward:.2f}, steps={t+1}")
    env.close()

if __name__ == "__main__":
    ENV_ID     = "highway-v0"
    MODEL_PATH = "pi_rec.pth"   # chemin vers votre modèle entraîné
    evaluate_pi_rec(ENV_ID, MODEL_PATH, num_episodes=5, max_steps=500)
