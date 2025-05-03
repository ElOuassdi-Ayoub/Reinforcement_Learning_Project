# safety_critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SafetyCriticClassifier(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        input_dim = state_dim + action_dim
        # réseau plus profond : 3 couches cachées
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)    # deux logits pour {no‐collision, collision}
        )

    def forward(self, states, actions_onehot):
        # concatène état et action one‐hot
        x = torch.cat([states, actions_onehot], dim=-1)
        logits = self.net(x)
        # on renvoie directement les logits ; appliquera softmax + CE en training
        return logits
