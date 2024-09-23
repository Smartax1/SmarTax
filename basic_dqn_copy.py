import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
         self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item() 
    # To implement irrationality
    # get probaility for each action, in act_value, exp neg * lambda, for action value i
    # prob(action_i) = exp(-lambda*act_value(i))/sum(x)
    # choose highest probabilty action's index

    def replay(self, batch_size):
        
        if len(self.memory) < batch_size:
            raise ValueError("Memory is empty. Cannot replay.")

        minibatch = random.sample(self.memory, batch_size)

        nstates, nactions, nrewards, nnext_states, ndones = [], [], [], [], []

        for i in range(len(minibatch)):  
            states, actions, rewards, next_states, dones = minibatch[i]
            nstates.append(states)
            nactions.append(actions)
            nrewards.append(rewards)
            nnext_states.append(next_states)
            ndones.append(dones)
        
        # Convert lists to numpy arrays
        nstates = np.array(nstates)
        nactions = np.array(nactions)
        nrewards = np.array(nrewards)
        nnext_states = np.array(nnext_states)
        ndones = np.array(ndones)

        # Convert lists to PyTorch tensors
        states = torch.FloatTensor(nstates)
        actions = torch.LongTensor(nactions)
        rewards = torch.FloatTensor(nrewards)
        next_states = torch.FloatTensor(nnext_states)
        dones = torch.FloatTensor(ndones)

        # Compute Q-values for the current states
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using the same model
        next_q_values = self.model(next_states).max(1)[0]

        # Compute the targets
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute the loss using the criterion
        criterion = nn.MSELoss()
        loss = criterion(q_values, targets)

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_checkpoint(self, filepath, episode, gov_rew, house_rew, epochs):
        torch.save({
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gov_rew': gov_rew,
            'house_rew': house_rew,
            'epochs': epochs
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        return checkpoint['episode'], checkpoint['gov_rew'], checkpoint['house_rew'], checkpoint['epochs']