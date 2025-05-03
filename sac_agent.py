import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import highway_env
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import trange

# ——— Hyper‐paramètres —————————————————————————————————
ENV_NAME   = "parking-v0"
EPISODES   = 500
STEPS      = 500
BATCH_SIZE = 256
REPLAY_SIZE= 100_000
GAMMA      = 0.99
TAU        = 0.005
ALPHA      = 0.2
LR         = 3e-4
WARMUP     = 1_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make(ENV_NAME, config={
    "observation": {"type": "Kinematics"},
    "action":      {"type": "ContinuousAction"},
    "duration": 50,
    "controlled_vehicles": 1,
    "vehicles_count":     1,
})
obs_shape = env.observation_space.shape       # ex. (5,5)
state_dim = int(np.prod(obs_shape))           # ici 25
action_dim= env.action_space.shape[0]         # ici 2
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, d):
        # s, s_next : vecteurs 1D de taille state_dim
        self.buffer.append((s, a, r, s_next, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return (
            torch.FloatTensor(s).to(device),                      # [B, state_dim]
            torch.FloatTensor(a).to(device),                      # [B, action_dim]
            torch.FloatTensor(r).unsqueeze(1).to(device),         # [B, 1]
            torch.FloatTensor(s_next).to(device),                 # [B, state_dim]
            torch.FloatTensor(d).unsqueeze(1).to(device)          # [B, 1]
        )

    def __len__(self):
        return len(self.buffer)

# ——— MLP basique —————————————————————————————————————————————
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# ——— Politique gaussienne tanh ———————————————————————————————
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.base       = MLP(state_dim, 2 * action_dim)
        self.action_dim = action_dim

    def forward(self, state):
        # state: [B, state_dim]
        out = self.base(state)  # [B, 2*action_dim]
        # split en deux moitiés le long de dim=1
        mu, log_std = out.split(self.action_dim, dim=1)
        log_std = torch.clamp(log_std, -20, 2)
        std     = log_std.exp()
        return mu, std

    def sample(self, state):
        # state: [B, state_dim]
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        x      = normal.rsample()            # [B, action_dim]
        a      = torch.tanh(x)
        # log_prob du vecteur d’action
        logp   = normal.log_prob(x).sum(dim=1, keepdim=True)   # [B,1]
        return a, logp

class SACAgent:
    def __init__(self):
        # réseaux
        self.policy     = GaussianPolicy(state_dim, action_dim).to(device)
        self.q1         = MLP(state_dim+action_dim, 1).to(device)
        self.q2         = MLP(state_dim+action_dim, 1).to(device)
        self.q1_targ    = MLP(state_dim+action_dim, 1).to(device)
        self.q2_targ    = MLP(state_dim+action_dim, 1).to(device)
        # copie des poids
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())
        # optimizers
        self.opt_pi  = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.opt_q1  = torch.optim.Adam(self.q1.parameters(), lr=LR)
        self.opt_q2  = torch.optim.Adam(self.q2.parameters(), lr=LR)

    def update(self, buf: ReplayBuffer):
        if len(buf) < BATCH_SIZE:
            return

        # 1) sample batch
        s, a, r, s2, done = buf.sample(BATCH_SIZE)  # shapes: [B,dim]

        # 2) compute Q target sous no_grad
        with torch.no_grad():
            a2, logp2 = self.policy.sample(s2)       # [B,act_dim], [B,1]
            # concat state2+action2
            inp2 = torch.cat([s2, a2], dim=1)         # [B,state_dim+act_dim]
            q1_t = self.q1_targ(inp2)                 # [B,1]
            q2_t = self.q2_targ(inp2)                 # [B,1]
            q_targ = r + GAMMA * (1.0 - done) * (
                      torch.min(q1_t, q2_t) - ALPHA * logp2
                    )                                   # [B,1]

        # 3) update Q1,Q2
        inp  = torch.cat([s, a], dim=1)                # [B,state_dim+act_dim]
        q1   = self.q1(inp)                            # [B,1]
        q2   = self.q2(inp)                            # [B,1]
        loss_q1 = F.mse_loss(q1, q_targ)
        loss_q2 = F.mse_loss(q2, q_targ)

        self.opt_q1.zero_grad()
        loss_q1.backward()
        self.opt_q1.step()

        self.opt_q2.zero_grad()
        loss_q2.backward()
        self.opt_q2.step()

        # 4) update policy π
        a_new, logp_new = self.policy.sample(s)        # [B,act_dim], [B,1]
        q1_new = self.q1(torch.cat([s, a_new], dim=1)) # [B,1]
        loss_pi = (ALPHA * logp_new - q1_new).mean()

        self.opt_pi.zero_grad()
        loss_pi.backward()
        self.opt_pi.step()

        # 5) soft‐update targets
        for targ, src in zip(self.q1_targ.parameters(), self.q1.parameters()):
            targ.data.mul_(1 - TAU)
            targ.data.add_(TAU * src.data)
        for targ, src in zip(self.q2_targ.parameters(), self.q2.parameters()):
            targ.data.mul_(1 - TAU)
            targ.data.add_(TAU * src.data)