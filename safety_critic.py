# =============================
# File: safety_critic.py
# =============================

import torch
import torch.nn as nn

class SafetyCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SafetyCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Pour un risque entre 0 et 1
        )

    def forward(self, state, action):
        # Action doit avoir même batch_size que state
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        x = torch.cat([state, action.float()], dim=-1)  # concaténation
        return self.net(x)
