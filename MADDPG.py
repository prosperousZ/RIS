import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
import matplotlib.pyplot as plt

class RISEnvironment(gym.Env):
    def __init__(self):
        super(RISEnvironment, self).__init__()
        self.N = 10  # Number of RIS elements
        self.action_space = spaces.MultiBinary(self.N)  # Phase shifts of 0 or pi
        self.observation_space = spaces.Box(low=0, high=1, shape=(2 * self.N,), dtype=np.float32)  # Real and imaginary parts
        self.max_steps = 50  # Maximum steps in an episode
        self.reset()

    def step(self, action):
        self.current_step += 1
        phases = np.pi * action
        Phi = np.diag(np.exp(1j * phases))
        received_signal = np.abs(np.dot(np.dot(self.h, Phi), self.g))**2
        reward = received_signal
        done = self.current_step >= self.max_steps
        next_state = self.get_state()
        return next_state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.h = np.random.randn(self.N) + 1j * np.random.randn(self.N)
        self.g = np.random.randn(self.N) + 1j * np.random.randn(self.N)
        return self.get_state()

    def get_state(self):
        state = np.concatenate([np.abs(self.h), np.abs(self.g)])
        return state

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

def train(env, actor, critic, actor_optimizer, critic_optimizer, episodes=500):
    episode_rewards = []
    for episode in range(episodes):
        state = torch.FloatTensor(env.reset()).unsqueeze(0)
        episode_reward = 0
        done = False
        while not done:
            action_probs = actor(state)
            action = torch.round(action_probs).detach().numpy().astype(int)[0]  # Binary action
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            episode_reward += reward

            action = torch.FloatTensor(action).unsqueeze(0)
            reward = torch.FloatTensor([reward]).unsqueeze(0)
            Q_value = critic(state, action)
            next_action_probs = actor(next_state)
            next_action = torch.round(next_action_probs)
            next_Q_value = critic(next_state, next_action.detach())
            expected_Q_value = reward + 0.99 * next_Q_value
            critic_loss = (Q_value - expected_Q_value.detach()).pow(2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_loss = -critic(state, action).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state
        episode_rewards.append(episode_reward)

    return episode_rewards

env = RISEnvironment()
actor = Actor(env.observation_space.shape[0], env.action_space.n)
critic = Critic(env.observation_space.shape[0], env.action_space.n)
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

rewards = train(env, actor, critic, actor_optimizer, critic_optimizer, episodes=100)

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards Over Time')
plt.show()
