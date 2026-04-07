import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid some "memory" errors with TkAgg backend
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_race
import torch
import torch.nn as nn
import torch.optim as optim

VERSION_NAME = 'DQN_v01' # the name for our model
REPORT_EPISODES  = 100 # report (plot) every...
DISPLAY_EPISODES = 50 # display live game every...

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def simulate(learning=True, episode_start=0):
    explore_rate = get_explore_rate(episode_start)
    total_rewards = []
    
    max_reward = -10_000
    env.set_view(True)

    for episode in range(episode_start, NUM_EPISODES + episode_start):
        if episode > 0:
            total_rewards.append(total_reward)

            if learning and episode % REPORT_EPISODES == 0:
                plt.plot(total_rewards)
                plt.ylabel('rewards')
                plt.show(block=False)
                plt.pause(4.0)
                file = f'models_{VERSION_NAME}/memory_{episode}'
                env.save_memory(file)
                
                model_file = f'models_{VERSION_NAME}/dqn_model_{episode}.pth'
                torch.save(policy_net.state_dict(), model_file)
                print(model_file, 'saved')
                plt.close()

        obv, _ = env.reset()
        state = obv
        total_reward = 0
        if not learning:
            env.pyrace.mode = 2

        if episode >= 1000:
            explore_rate = 0.01

        for t in range(MAX_T):
            action = select_action(state, explore_rate if learning else 0)
            obv, reward, done, _, info = env.step(action)
            next_state = obv
            
            env.remember(state, action, reward, next_state, done)
            memory.push(state, action, reward, next_state, done)
            
            total_reward += reward
            
            if learning:
                optimize_model()

            state = next_state
            
            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.set_msgs(['SIMULATE',
                            f'Episode: {episode}',
                            f'Time steps: {t}',
                            f'check: {info["check"]}',
                            f'dist: {info["dist"]}',
                            f'crash: {info["crash"]}',
                            f'Reward: {total_reward:.0f}',
                            f'Explore Rate: {explore_rate:.4f}',
                            f'Max Reward: {max_reward:.0f}'])
                env.render()
                
            if done or t >= MAX_T - 1:
                if total_reward > max_reward: max_reward = total_reward
                break
                
        explore_rate = get_explore_rate(episode)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
        
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    state_batch = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
    # the env returns discrete actions as a scalar e.g. 0,1,2 so batch_action is a tuple of ints
    action_batch = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1).to(device)
    next_state_batch = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
    done_batch = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1).to(device)

    # Q(s, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # max Q(s', a')
    with torch.no_grad():
        next_state_values = policy_net(next_state_batch).max(1)[0].unsqueeze(1)
        
    # Expected Q values
    expected_state_action_values = reward_batch + (DISCOUNT_FACTOR * next_state_values * (1 - done_batch))

    # Compute loss (MSE or Huber)
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping could be added here
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()

def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.max(1)[1].item()
    return action

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1.0, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def load_and_play(episode, learning=False):
    print("Start loading DQN model")
    model_file = f'models_{VERSION_NAME}/dqn_model_{episode}.pth'
    policy_net.load_state_dict(torch.load(model_file))
    policy_net.eval()
    
    # We could also load memory here if we want to continue learning
    if learning:
        policy_net.train()

    simulate(learning, episode)

if __name__ == "__main__":
    env = gym.make("Pyrace-v1").unwrapped # skip the TimeLimig and OrderEnforcing default wrappers
    print('env', type(env))
    if not os.path.exists(f'models_{VERSION_NAME}'): os.makedirs(f'models_{VERSION_NAME}')

    NUM_ACTIONS = env.action_space.n
    INPUT_DIM = env.observation_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MIN_EXPLORE_RATE  = 0.01
    DISCOUNT_FACTOR   = 0.99
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    
    DECAY_FACTOR = 16105.1 / 10.0

    NUM_EPISODES = 65_000
    MAX_T = 2000

    policy_net = DQN(INPUT_DIM, NUM_ACTIONS).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    memory = ReplayMemory(10000)

    #-------------
    simulate(learning=True) # LEARN starting from scratch...
    # load_and_play(100, learning=False) # e.g. run with loaded model
    #-------------
