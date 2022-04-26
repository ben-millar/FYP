from gym.spaces import Discrete, Box
import numpy as np

class KnapsackPacking(object):
    metadata = {}

    def __init__(self, num_knapsacks, capacity, min_item_size=1, max_item_size=10, min_item_value=1, max_item_value=100) -> None:
        self.capacity = capacity # Max capacity of a bin
        self.num_knapsacks = num_knapsacks
        self.min_item_size = min_item_size
        self.max_item_size = max_item_size
        self.min_item_value = min_item_value
        self.max_item_value = max_item_value

        # An array of bins with an integer representing their remaining capacity,
        # with one at the end representing the next item to be placed
        self.state = np.full(self.num_knapsacks + 1, self.capacity)
        self.values = np.full(self.num_knapsacks + 1, 0)

        self.getNewItem()

        high = np.full(self.state.size, self.capacity)
        low = np.full(self.state.size, 0)

        # Our observation space is each possible value of a bin
        self.observation_space = Box(low, high, dtype=np.int)
        
        # Our actions will index into our array of bins, or reject the item
        self.action_space = Discrete(len(self.state))
        
        self.reward_range = (-max_item_value, max_item_value)

        self.logs = { 'placed':0, 'misplaced':0, 'discarded':0 }


    def step(self, action):
        item_size = self.state[self.num_knapsacks]
        item_value = self.values[self.num_knapsacks]

        if action == len(self.state) - 1:
            # Discard item without trying to place it
            reward = 0
            # Generate a new item for the next step
            self.getNewItem()
            # Log event
            self.logs['discarded'] = self.logs['discarded'] + 1
        elif self.state[action] < item_size:
            # Attempted to place item in a bin that was too small
            reward = -10
            self.logs['misplaced'] = self.logs['misplaced'] + 1
        else:
            # Successfully placed item in a bin
            self.state[action] -= item_size 
            reward = item_value
            # Generate a new item for the next step
            self.getNewItem()
            self.logs['placed'] = self.logs['placed'] + 1

        # Returns true if all of our bins are at (or close to) capacity
        isTerminalState = np.all((self.min_item_size >= self.state))

        # Empty placeholder debug info dict
        info = {}

        return self.state, reward, isTerminalState, info


    def getNewItem(self):
        self.state[self.num_knapsacks] = \
            np.random.randint(low=self.min_item_size, high=self.max_item_size, size=1)

        self.values[self.num_knapsacks] = \
            np.random.randint(low=self.min_item_value, high=self.max_item_value, size=1)


    def render(self):
        pass


    def reset(self):
        self.state = np.full(self.num_knapsacks + 1, self.capacity)
        self.values = np.full(self.num_knapsacks + 1, 0)

        self.getNewItem()

        return self.state
