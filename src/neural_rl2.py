# Description: This file contains the neural network implementation of the RL algorithm
from collections import namedtuple, deque
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from kesslergame.Scenario_list import *
from kesslergame.Scenarios import *
from kesslergame.kessler_game_step import KesslerGameStep



class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return torch.tensor([480, 180], dtype=torch.float32) * torch.tanh(self.fc3(x))  # -1 to 1 action space


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append((Transition(*args)))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    # Hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 1e-3
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    MEMORY_SIZE = int(1e6)
    NUM_EPISODES = 500

    action_scale = torch.tensor([480, 180], dtype=torch.float32)

    # Initialize environment
    env = KesslerGameStep()
    state_size = 10
    action_size = 2  # Adjust based on your environment

    # Initialize actor and critic networks
    actor = Actor(state_size, action_size)
    critic = Critic(state_size, action_size)
    actor_target = Actor(state_size, action_size)
    critic_target = Critic(state_size, action_size)

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
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward).squeeze()
        done_batch = torch.stack(batch.done).squeeze()
        # Update critic
        critic_optimizer.zero_grad()

        actions_next = actor_target(non_final_next_states)
        # non_final_next_statesとactions_nextをcriticに入れる

        Q_targets_next = critic_target(non_final_next_states, actions_next)

        Q_targets_next = torch.squeeze(Q_targets_next)
        Q_targets = reward_batch + (GAMMA * Q_targets_next * (1 - done_batch))
        Q_expected = critic(state_batch, action_batch).squeeze()
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss.backward()
        critic_optimizer.step()

        # Update actor
        actor_optimizer.zero_grad()
        state_batch = torch.stack([state_batch])
        actions_pred = actor(state_batch)
        # actor_loss = -critic(state_batch, actions_pred).mean()

        actor_loss = -critic(state_batch[0], actions_pred[0]).mean()
        actor_loss.backward()
        actor_optimizer.step()

        # Soft update target networks
        soft_update(critic, critic_target, TAU)
        soft_update(actor, actor_target, TAU)

    #weight of rewards
    weight_r = torch.tensor([3.0, -1000, 0.1, 100], dtype=torch.float32)

    # Training loop
    for i_episode in range(1, NUM_EPISODES+1):
        state = env.reset(scenario=ring_static_top)
        rewards = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        print("episode:", i_episode)
        for t in itertools.count():
            action = actor(state).detach()

            next_state, reward_tag, terminated = env.run_step(action)

            reward = torch.dot(reward_tag, weight_r)
            rewards += reward_tag
            memory.push(state, action, next_state, reward, terminated)

            state = next_state

            optimize_model()

            if terminated:
                break
        print(f"hit={rewards[0]}, collision={rewards[1]}, survive frame = {rewards[2]}, survive to the end = {rewards[3]}")

    #save model
    torch.save(actor.state_dict(), 'model.pth')