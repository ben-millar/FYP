from gym.spaces import Discrete, Box
import numpy as np

class BinPacking2D(object):
    # Required by OpenAI Gym
    metadata = {}

    def __init__(self, num_bins, capacity, min_x=1, max_x=10, min_y=1, max_y=10) -> None:
        self.num_bins   = num_bins
        self.capacity   = capacity
        self.min_x      = min_x
        self.max_x      = max_x
        self.min_y      = min_y
        self.max_y      = max_y

        self.state = self.reset()

        low = np.hstack((
            np.full((self.num_bins + 1), 0),
            np.full((self.num_bins + 1), 0)
            )).reshape(self.num_bins + 1, 2)

        high = np.hstack((
            np.full((self.num_bins + 1), self.capacity),
            np.full((self.num_bins + 1), self.capacity)
            )).reshape(self.num_bins + 1, 2)

        self.observation_space = Box(
            low=low,
            high=high,
            shape=(self.num_bins+1, 2),
            dtype=np.int
        )

        # Actions 0-n select a bin, n+1 discards item
        self.action_space = Discrete(num_bins + 1)

        self.reward_range = (0, 10)


    def step(self, action):
        # Discard item
        if (action == self.num_bins):
            info = { 'placed':0, 'misplaced':0, 'discarded':1 }
            reward = 0
            self.getNewItem()
        else:
            new_item    = self.state[self.num_bins]
            chosen_bin  = self.state[action]

            remaining_capacity = chosen_bin - new_item

            # If this placement has caused any of our dimensions to go below 0
            if np.any((0 > remaining_capacity)):
                info = { 'placed':0, 'misplaced':1, 'discarded':0 }
                reward = 0
            else:
                info = { 'placed':1, 'misplaced':0, 'discarded':0 }
                reward = 10
                self.state[action] = remaining_capacity
                self.getNewItem()

        # Returns true if the x OR y capacities of all bins are below the minimum item sizes
        isTerminalState = \
            np.all(self.min_x >= self.state[:,0]) \
            or np.all(self.min_y >= self.state[:,1])

        return self.state, reward, isTerminalState, info


    def render(self):
        pass


    def reset(self):
        # An [n, 2] array representing the x,y capacities of each of our bins,
        # with an extra bin at the end representing the next item to be placed.
        self.state = np.hstack((
            np.full((self.num_bins + 1), self.capacity),
            np.full((self.num_bins + 1), self.capacity)
            )).reshape(self.num_bins + 1, 2)

        return self.state

    
    def getNewItem(self):
        x = np.random.randint(low=self.min_x, high=self.max_x)
        y = np.random.randint(low=self.min_y, high=self.max_y)
        self.state[self.num_bins] = [x,y]