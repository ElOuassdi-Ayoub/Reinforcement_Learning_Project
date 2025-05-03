import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from safety_critic import SafetyCriticClassifier
from dqn_agent import DQNAgent
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_offline = torch.load('D_offline.pt', weights_only=False)

state_dim  = len(D_offline[0][0])
action_dim = 5

# --- Critic binaire à 2 logits
critic = SafetyCriticClassifier(state_dim, action_dim).to(device)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
scheduler_critic = CosineAnnealingLR(optimizer_critic, T_max=200, eta_min=1e-5)

class_weights = torch.tensor([1.0, 10.0], device=device)
criterion     = torch.nn.CrossEntropyLoss(weight=class_weights)

# --- Policy de recovery supervisée
pi_rec = DQNAgent(state_dim, action_dim, device)
opt_rec = torch.optim.Adam(pi_rec.q_network.parameters(), lr=1e-3)
scheduler_rec = CosineAnnealingLR(opt_rec, T_max=200, eta_min=1e-5)
loss_rec_fn = torch.nn.CrossEntropyLoss()

# --- Hyperparamètres
BATCH_SIZE = 128
EPOCHS     = 200

# --- Stocks
loss_critic_history = []
loss_rec_history    = []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS} — lr_critic={scheduler_critic.get_last_lr()[0]:.4e}  lr_rec={scheduler_rec.get_last_lr()[0]:.4e}")
    random.shuffle(D_offline)
    pbar = tqdm(range(0, len(D_offline), BATCH_SIZE))

    epoch_losses_critic = []
    epoch_losses_rec    = []

    for i in pbar:
        batch = D_offline[i:i+BATCH_SIZE]
        states, actions, next_states, collisions = zip(*batch)

        # Tensors
        S = torch.tensor(np.array(states),     dtype=torch.float32, device=device)
        A = torch.tensor(actions,              dtype=torch.long,    device=device)
        C = torch.tensor(collisions,           dtype=torch.long,    device=device)  # 0 ou 1

        # one-hot actions
        A_onehot = torch.zeros(len(A), action_dim, device=device)
        A_onehot.scatter_(1, A.unsqueeze(1), 1.0)

        # --- Critic update (2 classes) ---
        logits = critic(S, A_onehot)           # [batch, 2]
        loss_critic = criterion(logits, C)     # CrossEntropy
        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()
        epoch_losses_critic.append(loss_critic.item())

        # --- Recovery policy (supervisé) ---
        with torch.no_grad():
            N = S.size(0)
            # On génère un batch agrandi pour toutes les actions
            S_exp = S.unsqueeze(1).repeat(1, action_dim, 1).view(N * action_dim, -1)
            A_exp = torch.eye(action_dim, device=device).unsqueeze(0).repeat(N, 1, 1)
            A_exp = A_exp.view(N * action_dim, action_dim)
            logits_all = critic(S_exp, A_exp)             # [N*act_dim, 2]
            probs_all  = F.softmax(logits_all, dim=1)     # [N*act_dim, 2]
            p_col = probs_all[:, 1].view(N, action_dim)   # probabilité de collision
            best_actions = torch.argmin(p_col, dim=1)     # action minimisant la collision

        logits_rec = pi_rec.q_network(S)  # [batch, action_dim]
        loss_rec   = loss_rec_fn(logits_rec, best_actions)
        opt_rec.zero_grad()
        loss_rec.backward()
        opt_rec.step()
        epoch_losses_rec.append(loss_rec.item())

        pbar.set_postfix({
            "L_critic": f"{loss_critic.item():.4f}",
            "L_pi_rec": f"{loss_rec.item():.4f}"
        })

    # Scheduler step en fin d(epoch)
    scheduler_critic.step()
    scheduler_rec.step()

    loss_critic_history.append(np.mean(epoch_losses_critic))
    loss_rec_history.append(np.mean(epoch_losses_rec))

# --- Affichage
plt.figure(figsize=(8,4))
plt.plot(loss_critic_history, label="Loss Critic (CE)")
plt.plot(loss_rec_history,    label="Loss π_rec")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# --- Sauvegarde
torch.save(critic.state_dict(),           'safety_critic_clf2.pth')
torch.save(pi_rec.q_network.state_dict(), 'pi_rec.pth')
print("✅ Critic (softmax prob) et π_rec entraînés avec CosineAnnealingLR.")
