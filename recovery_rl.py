import torch
import gymnasium as gym
import highway_env
import random
from dqn_agent import DQNAgent
from safety_critic import SafetyCritic

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("highway-v0", render_mode="human", config={
    "observation": {"type": "Kinematics", "vehicles_count": 5},
    "duration": 60,
    "vehicles_count": 30,
    "controlled_vehicles": 1,
    "lanes_count": 4,
})

state_dim = 25
action_dim = env.action_space.n

# Agents et critic
pi_task = DQNAgent(state_dim, action_dim, device)
pi_rec = DQNAgent(state_dim, action_dim, device)
pi_rec.q_network.load_state_dict(torch.load('pi_rec.pth'))

critic = SafetyCritic(state_dim, action_dim).to(device)
critic.load_state_dict(torch.load('safety_critic.pth'))
critic.train()

critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

# ✅ fonction de loss pondérée
def weighted_mse_loss(pred, target, weight):
    return (weight * (pred - target) ** 2).mean()

# Données
D_offline = torch.load('D_offline.pt', weights_only=False)
D_rec = D_offline.copy()
D_task = []

EPS_RISK = 0.001
GAMMA_RISK = 0.99
WEIGHT_FACTOR = 10  # pondération pour les collisions

# Training loop
for episode in range(1000):
    obs, info = env.reset()
    state = obs.flatten()
    done, truncated = False, False
    episode_reward = 0

    while not done and not truncated:
        action_task = pi_task.select_action(state)

        s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        a_one_hot = torch.zeros((1, action_dim), device=device)
        a_one_hot[0, action_task] = 1.0

        with torch.no_grad():
            q_risk = critic(s_tensor, a_one_hot)

        if q_risk.item() > EPS_RISK:
            action = pi_rec.select_action(state)
        else:
            action = action_task

        next_obs, reward, done, truncated, info = env.step(action)

        if episode % 10 == 0:
            env.render()

        next_state = next_obs.flatten()
        collision = int(info.get('crashed', False))

        D_task.append((state, action_task, next_state, reward))
        D_rec.append((state, action, next_state, collision))

        pi_task.store_transition(state, action, reward, next_state, done)
        pi_task.train_step()

        pi_rec.store_transition(state, action, -q_risk.item(), next_state, done)
        pi_rec.train_step()

        # ✅ Critic update avec pondération
        if len(D_rec) >= 64:
            batch = random.sample(D_rec, 64)
            states, actions, next_states, collisions = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            collisions = torch.FloatTensor(collisions).unsqueeze(1).to(device)
            actions = torch.LongTensor(actions).to(device)

            actions_onehot = torch.zeros((actions.size(0), action_dim), device=device)
            actions_onehot.scatter_(1, actions.unsqueeze(1), 1)

            with torch.no_grad():
                next_actions = torch.tensor(
                    [pi_task.select_action(ns.cpu().numpy()) for ns in next_states],
                    dtype=torch.long
                ).to(device)

                next_actions_onehot = torch.zeros((next_actions.size(0), action_dim), device=device)
                next_actions_onehot.scatter_(1, next_actions.unsqueeze(1), 1)

                next_q_risks = critic(next_states, next_actions_onehot)
                targets = collisions + (1 - collisions) * GAMMA_RISK * next_q_risks

            preds = critic(states, actions_onehot)

            # ✅ pondération des exemples avec collision
            weights = 1 + WEIGHT_FACTOR * collisions
            loss = weighted_mse_loss(preds, targets, weights)

            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()

        state = next_state
        episode_reward += reward

    print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Epsilon: {pi_task.epsilon:.2f}")

env.close()
print("✅ Recovery RL terminé avec critic pondéré (poids collisions amplifiés).")
