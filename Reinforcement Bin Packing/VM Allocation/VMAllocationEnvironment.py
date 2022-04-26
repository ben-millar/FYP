import csv

from gym.spaces import Discrete, Box
import numpy as np

class VMAllocationEnvironment(object):
    # Required by OpenAI Gym
    metadata = {}

    def __init__(self, num_bins, capacity) -> None:
        self.num_bins   = num_bins
        self.capacity   = capacity

        self.available_vms = []

        self.state = self.reset()

        low = np.hstack((
            np.full(self.num_bins + 1, 0),
            np.full(self.num_bins + 1, 0),
            np.full(self.num_bins + 1, 0),
            np.full(self.num_bins + 1, 0)
            )).reshape(self.num_bins, 4)

        high = np.hstack((
            np.full((self.num_bins + 1), self.capacity),
            np.full((self.num_bins + 1), self.capacity),
            np.full((self.num_bins + 1), self.capacity),
            np.full((self.num_bins + 1), self.capacity)
            )).reshape(self.num_bins + 1, 4)

        self.observation_space = Box(
            low=low,
            high=high,
            shape=(self.num_bins+1, 4),
            dtype=np.int
        )

        # Actions 0-n select a bin, n+1 discards item
        self.action_space = Discrete(num_bins + 1)

        self.reward_range = (-1, 1)

    
    def read_data(self):
        # Value | CPU | Memory | Storage | Popularity
        with open('real_data.csv') as csv_file:

            reader = csv.reader(csv_file, delimiter=',')

            skip_headers = True
    
            for row in reader:
                if skip_headers:
                    skip_headers = False
                    continue

                self.available_vms = row


    def step(self, action):
        # Discard item
        if (action == self.num_bins):
            info = { 'placed':0, 'misplaced':0, 'discarded':1 }
            reward = -1
            self.getNewItem()
        else:
            new_item    = self.state[self.num_bins]
            chosen_bin  = self.state[action]

            remaining_capacity = chosen_bin - new_item

            # If this placement has caused any of our dimensions to go below 0
            if np.any((0 > remaining_capacity)):
                info = { 'placed':0, 'misplaced':1, 'discarded':0 }
                reward = -new_item.prod() # Product of x*y
            else:
                info = { 'placed':1, 'misplaced':0, 'discarded':0 }
                reward = new_item.prod() # Product of x*y
                self.state[action] = remaining_capacity
                self.getNewItem()

        # Returns true if the x OR y capacities of all bins are below the minimum item sizes
        isTerminalState = np.all(self.state[:,0] <= self.min_x) or np.all(self.state[:,1] <= self.min_y)

        return self.state, reward, isTerminalState, info


    def render(self):
        pass


    def reset(self):
        # An [n, 4] array representing the value, CPU, Memory and Storage capacities
        # of each of our bins, with an extra bin at the end representing the next VM.
        self.state = np.hstack((
            np.full(self.num_bins + 1, 0), # Value
            np.full(self.num_bins + 1, self.capacity), # CPU
            np.full(self.num_bins + 1, self.capacity), # Memory
            np.full(self.num_bins + 1, self.capacity) # Storage
            )).reshape(self.num_bins + 1, 4)

        self.getNewItem()

        return self.state

    
    def getNewItem(self):
        index = np.random.randint(low=0, high=len(self.available_vms)-1)
        next_item = self.available_vms[index][:-1] # We ignore the final, popularity, value for now
        self.state[self.num_bins] = next_item