# sac_highway.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(
    "highway-v0",
    render_mode="human",
    config={
        "observation": {"type": "Kinematics", "vehicles_count": 5},
        "duration": 60,
        "vehicles_count": 30,
        "controlled_vehicles": 1,
        "lanes_count": 4,
    },
)

state_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n

class DiscreteSACPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs, logits

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.q_net(state)

policy = DiscreteSACPolicy(state_dim, action_dim).to(device)
q1 = QNetwork(state_dim, action_dim).to(device)
q2 = QNetwork(state_dim, action_dim).to(device)
alpha = torch.tensor(0.2, device=device, requires_grad=True)

policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
q1_opt = torch.optim.Adam(q1.parameters(), lr=3e-4)
q2_opt = torch.optim.Adam(q2.parameters(), lr=3e-4)
alpha_opt = torch.optim.Adam([alpha], lr=1e-4)
target_entropy = -np.log(1.0 / action_dim)

replay_buffer = deque(maxlen=100000)

# Hyperparams
BATCH_SIZE = 64
NUM_EPISODES = 100
MAX_STEPS = 500

for ep in range(1, NUM_EPISODES + 1):
    obs, _ = env.reset()
    state = obs.flatten().astype(np.float32)
    ep_reward = 0.0

    for _ in range(MAX_STEPS):
        s_t = torch.tensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, _, _ = policy(s_t)
        action = torch.distributions.Categorical(probs).sample().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = next_obs.flatten().astype(np.float32)
        done = terminated or truncated
        ep_reward += reward

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state

        if len(replay_buffer) >= BATCH_SIZE:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            s, a, r, s2, d = zip(*batch)
            S = torch.tensor(np.array(s), dtype=torch.float32, device=device)
            A = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
            R = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
            S2 = torch.tensor(np.array(s2), dtype=torch.float32, device=device)
            Dflag = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

            with torch.no_grad():
                probs_next, logp_next, _ = policy(S2)
                min_q_next = torch.min(q1(S2), q2(S2))
                V_next = (probs_next * (min_q_next - alpha * logp_next)).sum(dim=1, keepdim=True)
                target_q = R + (1 - Dflag) * 0.99 * V_next

            q1_pred = q1(S).gather(1, A)
            q2_pred = q2(S).gather(1, A)
            loss_q1 = F.mse_loss(q1_pred, target_q)
            loss_q2 = F.mse_loss(q2_pred, target_q)

            q1_opt.zero_grad(); loss_q1.backward(); q1_opt.step()
            q2_opt.zero_grad(); loss_q2.backward(); q2_opt.step()

            probs, logp, _ = policy(S)
            min_q_pi = torch.min(q1(S), q2(S))
            policy_loss = (probs * (alpha * logp - min_q_pi)).sum(dim=1).mean()

            policy_opt.zero_grad(); policy_loss.backward(); policy_opt.step()

            entropy = - (probs * logp).sum(dim=1).mean()
            alpha_loss = -(alpha * (entropy - target_entropy).detach()).mean()

            alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

        if done:
            break

    print(f"[Ep {ep}/{NUM_EPISODES}] Reward = {ep_reward:.1f}")

env.close()
print("✅ SAC classique terminé.")
