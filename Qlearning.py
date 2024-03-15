#First of all, this is single-input, single output scheme
#h and g are fading channels between single-antenna source and the RIS
#for the ith reflecting meta-surface antenna (i = 1,2,...,N) and N is the numebr of reflecting metasurface of the RIS
#Under the assumption of Rayleigh fading channels,
#path loss not considered
#h and g are random generate

#*****************************************************
#action: RIS unit phase shift
#state: RIS current phase shift config, the vector of phase shift for every unit.(state需要考虑信道，state must consider the channel）
#Reward: based one action, received signal strength, signal strength increase, reward increase.
#*****************************************************


# 1. define channel inside the state
# 2. find the paper for that optimization, then send the paper
# 3. write the optimization solution code, then compare with the Q-learning.

#for this code, it does not consider how x was adjusted, we simply data symbol = 1
#for this code, every RIS unit that has two state(0 or pi), and every unit can adjust seperately.

import numpy as np
import matplotlib.pyplot as plt

class RISEnvironment:
    #all the parameter in this environment could be change or recaculated by more specific channel information
    def __init__(self, num_elements=10):
        self.num_elements = num_elements# RIS 单元数量
        
        #variance = 1,channel gain for both h and g set up = 1, can be changed any time
        self.sigma_alpha = 1
        self.sigma_beta = 1
        
        # generate Raylrigh distributed alpha and beta
        self.alpha =np.random.rayleigh(self.sigma_alpha, num_elements)
        self.beta = np.random.rayleigh(self.sigma_beta, num_elements)
        
        #generate theta and psi
        self.theta = np.random.rand(num_elements) * 2 * np.pi
        self.psi = np.random.rand(num_elements) * 2 * np.pi
       
        #initial Es = 1 No = 1, so fixed Es/N0
        self.Es = 1
        self.N0 = 1
        
        # 初始化信道系数向量 h 和 g
        self.h = self.alpha * np.exp(-1j * self.theta)
        self.g = self.beta * np.exp(-1j * self.psi)

        #data symbol
        self.x = 1

        #self.AWGN = np.random.normal(0, np.sqrt(self.N0), 1) + 1j * np.random.normal(0, np.sqrt(self.N0), 1)
        #self.h = np.random.normal(0, 1, num_elements) + 1j * np.random.normal(0, 1, num_elements)
        #self.g = np.random.normal(0, 1, num_elements) + 1j * np.random.normal(0, 1, num_elements)
        self.noise_power = 0.01
        self.state = np.zeros(num_elements, dtype=int) # 初始化RIS相位状态

    def step(self, action):
        # Convert action to phase state for each RIS element
        # 更新RIS相位状态,目前state只跟action相关，还没有加入channel
        # this line of code represent a integer 0 or 1 action array
        self.state = [int(x) for x in format(action, '0{}b'.format(self.num_elements))]
        received_signal, reward = self.calculate_received_signal_and_reward()
        return self.encode_state(), reward

    #calculate the SNR and reward(reward is gamma SNR)
    def calculate_received_signal_and_reward(self):
        phase_shifts = np.pi * np.array(self.state)
        Phi = np.exp(1j * phase_shifts)

        #This line of code corresponding to equation 9
        effective_channel = np.sum(self.h * Phi * self.g) * self.x
        
        #noise here, gaussian noise ~ N(0,0,1)
        self.noise = np.random.normal(scale=np.sqrt(self.noise_power))
        #noise = self.AWGN
        
        received_signal = effective_channel + self.noise
        signal_power = np.abs(received_signal) ** 2
        snr = signal_power / self.noise_power
        reward = snr
        return received_signal, reward

    def encode_state(self):
        return int(''.join(map(str, self.state)), 2)

    def reset(self):
        self.state = np.zeros(self.num_elements, dtype=int)
        return self.encode_state()
    
    #This method is to maximized by eliminating the channel phase, then compare with Q learning
    def calculate_maximum_snr(self):
        phase_shifts = self.theta + self.psi
        Phi = np.exp(1j * phase_shifts)
        #This line of code corresponding to equation 9
        effective_channel = np.sum(self.h * Phi * self.g) * self.x
        received_signal = effective_channel + self.noise
        signal_power = np.abs(received_signal) ** 2
        snr = signal_power / self.noise_power
        return snr

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.action_space_size = action_space_size

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(0, self.action_space_size)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        next_max = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_rate * next_max - self.q_table[state, action])

    def decay_exploration_rate(self):
        self.exploration_rate *= 0.995
        self.exploration_rate = max(self.exploration_rate, 0.01)

# Initialize environment and agent
num_elements = 10
state_space_size = 2 ** num_elements
action_space_size = 2 ** num_elements
env = RISEnvironment(num_elements=num_elements)
agent = QLearningAgent(state_space_size, action_space_size)

# Run training
num_episodes = 2000
rewards = []

snr_compare = []
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    total_snr = 0

    for _ in range(100):  # Limit the number of steps per episode
        action = agent.choose_action(state)
        next_state, reward = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        snr_result = env.calculate_maximum_snr()
        total_snr += snr_result

    agent.decay_exploration_rate()
    rewards.append(total_reward)
    snr_compare.append(total_snr)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

# Plotting the training progress
plt.figure("1")
plt.plot(rewards)
plt.xlabel('slot')
plt.ylabel('Total snr')
plt.title('Training Progress')
plt.show()
plt.figure("2")
plt.plot(snr_compare)
plt.xlabel('slot')
plt.ylabel('Total snr')
plt.title('Training Progress')
plt.show()
