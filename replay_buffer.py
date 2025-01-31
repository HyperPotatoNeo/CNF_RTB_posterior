import torch
from collections import deque
import random
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, rb_size=1000, rb_sample_strategy='uniform', rb_beta=1.0):
        self.buffer_size = rb_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.rewards = deque(maxlen=self.buffer_size)
        self.losses = deque(maxlen=self.buffer_size)
        self.sample_strategy = rb_sample_strategy
        self.beta = rb_beta

    def add(self, image, reward, loss):
        self.buffer.extend(torch.unbind(image.cpu(), dim=0))
        self.rewards.extend(reward.cpu().numpy().tolist())
        self.losses.extend(loss.cpu().numpy().tolist())

    def sample_uniform(self, batch_size):
        batch_indices = np.random.randint(len(self.buffer), size=batch_size).tolist()
        batch = [self.buffer[idx] for idx in batch_indices]
        reward_batch = [self.rewards[idx] for idx in batch_indices]
        return torch.stack(batch).to(DEVICE), torch.tensor(reward_batch).to(DEVICE) 

    def sample_reward(self, batch_size):
        ## Sample proportional to exp(r(x)*beta)
        logZ = torch.logsumexp(torch.tensor(self.rewards)*self.beta, dim=0).item()
        probabilities = [np.exp(reward*self.beta - logZ) for reward in self.rewards]
        batch_indices = random.choices(range(len(self.buffer)), weights=probabilities, k=batch_size)
        batch = [self.buffer[idx] for idx in batch_indices]
        reward_batch = [self.rewards[idx] for idx in batch_indices]
        return torch.stack(batch).to(DEVICE), torch.tensor(reward_batch).to(DEVICE)
    
    def sample_loss(self, batch_size):
        ## Sample proportional to loss
        ## TODO
        pass
    
    def sample(self, batch_size):
        if self.sample_strategy == 'uniform':
            batch, reward_batch = self.sample_uniform(batch_size)
            return batch, reward_batch
        elif self.sample_strategy == 'reward':
            batch, reward_batch = self.sample_reward(batch_size)
            batch_unif, reward_batch_unif = self.sample_uniform(batch_size)

            # sample with 1/4 high reward samples, rest uniform reward
            
            num_hi_reward = batch_size//4 
            
            print("\nBuffer Num hi reward: ", num_hi_reward)
            batch[num_hi_reward:] = batch_unif[num_hi_reward:] 
            reward_batch[num_hi_reward:] = reward_batch_unif[num_hi_reward:]

            return batch, reward_batch
        else:
            print('INVALID SAMPLE STRATEGY')
            exit()

    def size(self):
        return len(self.buffer)