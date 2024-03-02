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
        # Number of RIS elements
        self.N = 10
        # Phase shifts of 0 or pi
        self.action_space = spaces.MultiBinary(self.N)  
        self.observation_space = spaces.Box(low=0, high=1, shape=(2 * self.N,), dtype=np.float32)  # Real and imaginary parts
        self.max_steps = 50  # Maximum steps in an episode
        self.reset()

    def step(self, action):
        #increment step count
        self.current_step += 1
        #convert actions to phase shift
        phases = np.pi * action

        #*************
        #construct the phase shift matrix
        #*************
        Phi = np.diag(np.exp(1j * phases))

        #calculate received signal
        received_signal = np.abs(np.dot(np.dot(self.h, Phi), self.g))**2
        #set the reward = receive signal
        reward = received_signal

        #set the end condition
        done = self.current_step >= self.max_steps
        next_state = self.get_state()
        return next_state, reward, done, {}

    #reset method, to reset the state
    def reset(self):
        #reset step count at the start of an episode
        self.current_step = 0
        #initialize rayleigh channel coefficients randomly
        self.h = (np.random.randn(self.N) + 1j * np.random.randn(self.N))/np.sqrt(2)
        self.g = (np.random.randn(self.N) + 1j * np.random.randn(self.N))/np.sqrt(2)
        return self.get_state()

    #return the state
    def get_state(self):
        state = np.concatenate([np.abs(self.h), np.abs(self.g)])
        return state

#actor network
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        #define a simple neural network with on hidden layer
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            #sigmoid to output probabilities for binary actions
            nn.Sigmoid()
        )
        
    #forward
    def forward(self, state):
        #forward pass through the network to evaluate state-action pairs
        return self.network(state)

#critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        #Define a simple neural network with one hidden layer
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    #forward methold
    def forward(self, state, action):
        #forward pass through the network to evaluate state-action pairs
        return self.network(torch.cat([state, action], dim=1))

def train(env, actor, critic, actor_optimizer, critic_optimizer, episodes=500):
    episode_rewards = []
    for episode in range(episodes):
        state = torch.FloatTensor(env.reset()).unsqueeze(0)
        episode_reward = 0
        done = False
        while not done:
            #actor decides on the action based on the current state
            action_probs = actor(state)
            action = torch.round(action_probs).detach().numpy().astype(int)[0]  # Binary action
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            episode_reward += reward

            #prepare tensors for critic update
            action = torch.FloatTensor(action).unsqueeze(0)
            reward = torch.FloatTensor([reward]).unsqueeze(0)
            Q_value = critic(state, action)
            next_action_probs = actor(next_state)
            next_action = torch.round(next_action_probs)
            next_Q_value = critic(next_state, next_action.detach())
            expected_Q_value = reward + 0.99 * next_Q_value
            critic_loss = (Q_value - expected_Q_value.detach()).pow(2).mean()

            #update critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            #update actor
            actor_loss = -critic(state, action).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state
        episode_rewards.append(episode_reward)

    return episode_rewards

#initialize environment and networks
env = RISEnvironment()
actor = Actor(env.observation_space.shape[0], env.action_space.n)
critic = Critic(env.observation_space.shape[0], env.action_space.n)
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

rewards = train(env, actor, critic, actor_optimizer, critic_optimizer, episodes=100)

#plot the reward with episodes
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards Over Time')
plt.show()
