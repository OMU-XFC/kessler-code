
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Controller_neuro import Controller_neuro
from kessler_game.new.Scenario_list import ring_static_top
from kessler_game.src.kesslergame.kessler_game_step import KesslerGameStep as Game_step


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        state = state.unsqueeze_(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # -1 to 1 action space


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        print("Critic_forward")
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
MEMORY_SIZE = int(1e6)
NUM_EPISODES = 600

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment
env = Game_step()
state_size = len(env.reset(scenario=ring_static_top, ))
action_size = 2  # Adjust based on your environment

# Initialize actor and critic networks
actor = Actor(state_size, action_size).to(device)
critic = Critic(state_size, action_size).to(device)
actor_target = Actor(state_size, action_size).to(device)
critic_target = Critic(state_size, action_size).to(device)

# Initialize target networks
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

# Initialize optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

# Initialize replay memory
memory = ReplayMemory(MEMORY_SIZE)


def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    done_batch = torch.cat(batch.done)

    # Update critic
    critic_optimizer.zero_grad()
    print(non_final_next_states.shape)
    actions_next = actor_target(non_final_next_states)
    Q_targets_next = critic_target(non_final_next_states, actions_next).detach()
    Q_targets = reward_batch + (GAMMA * Q_targets_next * (1 - done_batch))
    Q_expected = critic(state_batch, action_batch)
    critic_loss = F.mse_loss(Q_expected, Q_targets)
    critic_loss.backward()
    critic_optimizer.step()

    # Update actor
    actor_optimizer.zero_grad()
    actions_pred = actor(state_batch)
    actor_loss = -critic(state_batch, actions_pred).mean()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft update target networks
    soft_update(critic, critic_target, TAU)
    soft_update(actor, actor_target, TAU)


# Training loop
for i_episode in range(NUM_EPISODES):
    state = env.reset(scenario=ring_static_top)
    state = torch.tensor(state, dtype=torch.float32, device=device).clone().detach()
    for t in itertools.count():
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = actor(state).detach()
        next_state, reward, terminated, truncated, _ = env.run_step(action)
        memory.push(state, action, next_state, reward, terminated or truncated)

        state = next_state

        optimize_model()

        if terminated or truncated:
            break
