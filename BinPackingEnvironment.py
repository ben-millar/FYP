from gym.spaces import Discrete, Box
import numpy as np
import matploylib.pyplot as plt
import random

class BinPacking(object):
    def __init__(self, num_bins, capacity) -> None:
        self.capacity = capacity # Max capacity of a bin
        self.num_bins = num_bins

        # An array of bins with an integer representing their remaining capacity
        self.state = np.full((self.num_bins), self.capacity)
        
        # Our actions will index into our array of bins
        self.action_space = Discrete(len(self.state))


    def step(self, item_size, action):
        # Generate a random integer between 1 and our max capacity
        #new_item = np.random.randint(low=1, high=self.capacity, size=1)
        if self.state[action] < item_size:
            reward = 0
        else:
            self.state[action] -= item_size 
            reward = item_size

        # Returns true if all of our bins are at 0 capacity
        isTerminalState = np.all((0 == self.state))

        # Empty placeholder debug info dict
        info = {}

        return self.state, reward, isTerminalState, info

    def render(self):
        pass

    def reset(self):
        self.state = np.full((self.num_bins), self.capacity)
        return self.state
