# recovery_rl_sac.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import highway_env
import random
import numpy as np
import time
from collections import deque
from safety_critic import SafetyCriticClassifier
from dqn_agent import DQNAgent

# --- Setup général
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

# --- Recharge pi_rec (DQN)
pi_rec = DQNAgent(state_dim, action_dim, device)
pi_rec.q_network.load_state_dict(torch.load("pi_rec.pth", map_location=device))
pi_rec.q_network.to(device).train()

# --- Recharge le Safety Critic
critic = SafetyCriticClassifier(state_dim, action_dim).to(device)
critic.load_state_dict(torch.load("safety_critic_clf2.pth", map_location=device), strict=False)
critic.train()
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

# === SAC pi_task (Discrete)
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

# --- Hyperparamètres
EPS_RISK = 0.9
GAMMA_RISK = 0.99
WEIGHT_FACTOR = 10
BATCH_SIZE = 64
NUM_EPISODES = 50

# --- Chargement de D_offline
from numpy import ndarray
torch.serialization.add_safe_globals([ndarray])
raw_off = torch.load("D_offline.pt", map_location="cpu", weights_only=False)

D_offline = deque(raw_off, maxlen=100000)
D_rec = deque(maxlen=100000)
for item in raw_off:
    if len(item) == 4:
        s, a, s2, d_or_r = item
        if isinstance(d_or_r, (bool, np.bool_)):
            D_rec.append((s, a, 0, s2, d_or_r))
        else:
            D_rec.append((s, a, 0, s2, False))
    elif len(item) == 5:
        s, a, r, s2, d = item
        D_rec.append((s, a, 0, s2, d))
    else:
        raise ValueError(f"Format inattendu : {item}")
D_task = deque(maxlen=100000)

def weighted_mse_loss(pred, target, weight):
    return (weight * (pred - target)**2).mean()

# === Entraînement
for ep in range(1, NUM_EPISODES + 1):
    obs, info = env.reset()
    state = obs.flatten().astype(np.float32)
    done = False
    ep_reward = 0.0

    while not done:
        s_t = torch.from_numpy(state).unsqueeze(0).to(device)
        probs, _, _ = policy(s_t)
        dist = torch.distributions.Categorical(probs)
        a_task = dist.sample().item()

        a1hot = torch.zeros(1, action_dim, device=device)
        a1hot[0, a_task] = 1.0
        with torch.no_grad():
            q_logits = critic(s_t, a1hot)
            q_risk = F.softmax(q_logits, dim=1)[0, 1]

        if q_risk.item() > EPS_RISK:
            action = pi_rec.select_action(state)
        else:
            action = a_task

        next_obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.03)
        done = terminated or truncated
        next_state = next_obs.flatten().astype(np.float32)
        collision = int(info.get("crashed", False))

        ep_reward += reward

        D_task.append((state, action, reward, next_state, done))
        D_rec.append((state, action, collision, next_state, done))

        # === Update SAC
        if len(D_task) >= BATCH_SIZE:
            batch = random.sample(D_task, BATCH_SIZE)
            s, a, r, s2, d = zip(*batch)
            S = torch.tensor(np.array(s), dtype=torch.float32, device=device)
            A = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
            R = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
            S2 = torch.tensor(np.array(s2), dtype=torch.float32, device=device)
            Dflag = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

            with torch.no_grad():
                probs_next, logp_next, _ = policy(S2)
                q1_next = q1(S2)
                q2_next = q2(S2)
                min_q_next = torch.min(q1_next, q2_next)
                V_next = (probs_next * (min_q_next - alpha * logp_next)).sum(dim=1, keepdim=True)
                target_q = R + (1 - Dflag) * 0.99 * V_next

            q1_pred = q1(S).gather(1, A)
            q2_pred = q2(S).gather(1, A)
            loss_q1 = F.mse_loss(q1_pred, target_q)
            loss_q2 = F.mse_loss(q2_pred, target_q)

            q1_opt.zero_grad()
            loss_q1.backward()
            q1_opt.step()

            q2_opt.zero_grad()
            loss_q2.backward()
            q2_opt.step()

            probs, logp, _ = policy(S)
            q1_pi = q1(S)
            q2_pi = q2(S)
            min_q_pi = torch.min(q1_pi, q2_pi)
            policy_loss = (probs * (alpha * logp - min_q_pi)).sum(dim=1).mean()

            policy_opt.zero_grad()
            policy_loss.backward()
            policy_opt.step()

            entropy = - (probs * logp).sum(dim=1).mean()
            alpha_loss = -(alpha * (entropy - target_entropy).detach()).mean()

            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

        state = next_state

    print(f"[Ep {ep}/{NUM_EPISODES}] Reward={ep_reward:.1f}")

env.close()
print("\u2705 Recovery RL avec SAC terminé.")
