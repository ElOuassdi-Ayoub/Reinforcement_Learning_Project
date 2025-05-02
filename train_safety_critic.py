import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from safety_critic import SafetyCritic
from dqn_agent import DQNAgent

# === Setup général
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Charger D_offline
D_offline = torch.load('D_offline.pt', weights_only=False)

state_dim = len(D_offline[0][0])
action_dim = 5  # Pour highway-v0

# === Critic
critic = SafetyCritic(state_dim, action_dim).to(device)
critic.train()
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

# ✅ Fonction de perte pondérée
def weighted_mse_loss(pred, target, weight):
    return (weight * (pred - target) ** 2).mean()

# === Policy pi_rec
pi_rec = DQNAgent(state_dim, action_dim, device)
pi_rec_optimizer = torch.optim.Adam(pi_rec.q_network.parameters(), lr=1e-3)
pi_rec_loss_fn = torch.nn.CrossEntropyLoss()

# === Hyperparamètres
BATCH_SIZE = 128
NUM_EPOCHS = 20
GAMMA_RISK = 0.99
WEIGHT_FACTOR = 10  # poids pour c=1

# === Suivi des pertes
critic_losses = []
pi_rec_losses = []

# === Boucle d'entraînement
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    random.shuffle(D_offline)
    progress_bar = tqdm(range(0, len(D_offline), BATCH_SIZE), desc=f"Training Epoch {epoch+1}")

    epoch_critic_loss = []
    epoch_pi_rec_loss = []

    for i in progress_bar:
        batch = D_offline[i:i+BATCH_SIZE]
        states, actions, next_states, collisions = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        collisions = torch.FloatTensor(collisions).unsqueeze(1).to(device)
        actions = torch.LongTensor(actions).to(device)

        # ======== Entraînement du Critic ========
        with torch.no_grad():
            next_actions_list = []
            for ns in next_states:
                q_values = []
                for a in range(action_dim):
                    action_one_hot = torch.zeros((1, action_dim), device=device)
                    action_one_hot[0, a] = 1.0
                    q = critic(ns.unsqueeze(0), action_one_hot)
                    q_values.append(q.item())
                best_action = q_values.index(min(q_values))
                next_actions_list.append(best_action)

            next_actions = torch.tensor(next_actions_list, dtype=torch.long, device=device)
            next_actions_onehot = torch.zeros((next_actions.size(0), action_dim), device=device)
            next_actions_onehot.scatter_(1, next_actions.unsqueeze(1), 1)

            next_q_risks = critic(next_states, next_actions_onehot)
            targets = collisions + (1 - collisions) * GAMMA_RISK * next_q_risks

        actions_onehot = torch.zeros((actions.size(0), action_dim), device=device)
        actions_onehot.scatter_(1, actions.unsqueeze(1), 1)

        preds = critic(states, actions_onehot)

        # ✅ pondérer fortement les c=1
        weights = 1 + WEIGHT_FACTOR * collisions
        loss_critic = weighted_mse_loss(preds, targets, weights)

        critic_optimizer.zero_grad()
        loss_critic.backward()
        critic_optimizer.step()

        epoch_critic_loss.append(loss_critic.item())

        # ======== Entraînement de pi_rec (policy supervisée)
        with torch.no_grad():
            all_qs = []
            for s in states:
                q_values = []
                for a in range(action_dim):
                    action_one_hot = torch.zeros((1, action_dim), device=device)
                    action_one_hot[0, a] = 1.0
                    q = critic(s.unsqueeze(0), action_one_hot)
                    q_values.append(q.item())
                all_qs.append(q_values)

            best_actions = torch.tensor(
                [qs.index(min(qs)) for qs in all_qs],
                dtype=torch.long, device=device
            )

        preds_rec = pi_rec.q_network(states)
        loss_rec = pi_rec_loss_fn(preds_rec, best_actions)

        pi_rec_optimizer.zero_grad()
        loss_rec.backward()
        pi_rec_optimizer.step()

        epoch_pi_rec_loss.append(loss_rec.item())

    critic_losses.append(np.mean(epoch_critic_loss))
    pi_rec_losses.append(np.mean(epoch_pi_rec_loss))

# === Affichage des courbes
plt.figure(figsize=(10, 5))
plt.plot(critic_losses, label="Critic Loss (pondérée)")
plt.plot(pi_rec_losses, label="Recovery Policy Loss (pi_rec)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Évolution des Losses pendant l'entraînement")
plt.legend()
plt.grid()
plt.show()

# === Sauvegarde
torch.save(critic.state_dict(), 'safety_critic.pth')
torch.save(pi_rec.q_network.state_dict(), 'pi_rec.pth')
print("✅ Critic et π_rec entraînés avec pondération (balancement de classes).")
