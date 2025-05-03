import gymnasium as gym
import highway_env
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

# ⚙️ Paramètres
NUM_EPISODES = 1000
MAX_STEPS = 500
MOMENTUM_STEPS = 4

# ⚙️ Initialisation
env = gym.make("highway-v0", render_mode=None, config={
    "observation": {"type": "Kinematics", "vehicles_count": 5},
    "duration": 40,
    "vehicles_count": 30,
    "controlled_vehicles": 1,
    "lanes_count": 4,
})

D_offline = []

# ✅ Fonction de comparaison correcte
def is_in_momentum(transition, momentum_queue):
    state, action, next_state = transition
    for m_state, m_action, m_next_state in momentum_queue:
        if (
            np.array_equal(state, m_state)
            and action == m_action
            and np.array_equal(next_state, m_next_state)
        ):
            return True
    return False

# 📦 Génération
for ep in tqdm(range(NUM_EPISODES), desc="Collecting D_offline"):
    obs, info = env.reset()
    state = obs.flatten()
    done, truncated = False, False

    episode_transitions = []
    momentum_queue = deque(maxlen=MOMENTUM_STEPS)
    collision_detected = False

    for _ in range(MAX_STEPS):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        next_state = next_obs.flatten()
        collision = int(info.get('crashed', False))

        transition = (state, action, next_state)
        episode_transitions.append(transition)
        momentum_queue.append(transition)

        if collision == 1:
            collision_detected = True
            break

        state = next_state

        if done or truncated:
            break

    # === Après l'épisode ===
    if collision_detected:
        for transition in episode_transitions:
            if is_in_momentum(transition, momentum_queue):
                D_offline.append((*transition, 1))  # Proche collision
            else:
                D_offline.append((*transition, 0))  # Normal
    else:
        for transition in episode_transitions:
            D_offline.append((*transition, 0))

# 💾 Sauvegarde
torch.save(D_offline, 'D_offline.pt')
print(f"✅ D_offline saved with {len(D_offline)} transitions (c=0 normal, c=1 momentum)")
