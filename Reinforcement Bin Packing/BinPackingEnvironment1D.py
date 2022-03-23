from gym.spaces import Discrete, Box
import numpy as np

class BinPacking(object):
    metadata = {}

    def __init__(self, num_bins, capacity, min_item_size) -> None:
        self.capacity = capacity # Max capacity of a bin
        self.num_bins = num_bins
        self.min_item_size = min_item_size

        # An array of bins with an integer representing their remaining capacity
        self.state = np.full((self.num_bins + 1), self.capacity)

        high = np.full(self.state.size, self.capacity)
        low = np.full(self.state.size, 0)

        # Our observation space is each possible value of a bin
        self.observation_space = Box(low, high, dtype=np.int)
        
        # Our actions will index into our array of bins, or reject the item
        self.action_space = Discrete(len(self.state))
        
        self.reward_range = (-1, 1)

        self.logs = { 'placed':0, 'misplaced':0, 'discarded':0 }


    def step(self, action):
        item_size = self.state[self.num_bins]

        if action == len(self.state) - 1:
            # Discard item without trying to place it
            reward = 0
            # Generate a new item for the next step
            self.getNewItem()
            # Log event
            self.logs['discarded'] = self.logs['discarded'] + 1
        elif self.state[action] < item_size:
            # Attempted to place item in a bin that was too small
            reward = -1
            self.logs['misplaced'] = self.logs['misplaced'] + 1
        else:
            # Successfully placed item in a bin
            self.state[action] -= item_size 
            reward = 1
            # Generate a new item for the next step
            self.getNewItem()
            self.logs['placed'] = self.logs['placed'] + 1

        # Returns true if all of our bins are at (or close to) capacity
        isTerminalState = np.all((self.min_item_size > self.state))

        # Empty placeholder debug info dict
        info = {}

        return self.state, reward, isTerminalState, info


    def getNewItem(self):
        self.state[self.num_bins] = \
            np.random.randint(low=self.min_item_size, high=self.capacity/2, size=1)


    def render(self):
        pass


    def reset(self):
        self.state = np.full((self.num_bins + 1), self.capacity)
        self.getNewItem()
        return self.state
