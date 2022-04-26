import csv
from tokenize import Double

from gym.spaces import Discrete, Box
import numpy as np

class VMAllocationEnvironment(object):
    # Required by OpenAI Gym
    metadata = {}

    def __init__(self, num_bins, capacity) -> None:
        self.num_bins   = num_bins
        self.capacity   = capacity

        self.available_vms = []
        self.read_data()

        self.state = self.reset()

        low = np.column_stack((
            np.full(self.num_bins + 1, np.float32(0)),
            np.full(self.num_bins + 1, np.float32(0)),
            np.full(self.num_bins + 1, np.float32(0)),
            np.full(self.num_bins + 1, np.float32(0))
            ))

        high = np.column_stack((
            np.full((self.num_bins + 1), np.float32(100)),
            np.full((self.num_bins + 1), np.float32(self.capacity)),
            np.full((self.num_bins + 1), np.float32(self.capacity)),
            np.full((self.num_bins + 1), np.float32(self.capacity))
            ))

        self.observation_space = Box(
            low=low,
            high=high,
            shape=(self.num_bins+1, 4)
        )

        # Actions 0-n select a bin, n+1 discards item
        self.action_space = Discrete(num_bins + 1)

        self.reward_range = (-1, 1)

    
    def read_data(self):
        # Value | CPU | Memory | Storage | Popularity
        with open('real_data.csv') as csv_file:

            reader = csv.reader(csv_file, delimiter=',')

            skip_header = True

            for row in reader:
                if skip_header:
                    skip_header = False
                    continue

                self.available_vms.append([float(i) for i in row])


    def step(self, action):
        # Discard item
        if (action == self.num_bins):
            info = { 'placed':0, 'misplaced':0, 'discarded':1 }
            reward = -new_item[0]
            self.getNewItem()
        else: # Place item
            new_item    = self.state[self.num_bins]
            chosen_bin  = self.state[action]

            # Value is +ve, sizes are -ve
            remaining_capacity = chosen_bin + new_item

            # If this placement has caused any of our dimensions to go below 0
            if np.any((0 > remaining_capacity)):
                info = { 'placed':0, 'misplaced':1, 'discarded':0 }
                reward = -1
            else:
                info = { 'placed':1, 'misplaced':0, 'discarded':0 }
                reward = new_item[0] # Value
                self.state[action] = remaining_capacity
                self.getNewItem()

        # Returns true if none of our bins have any capacity for any one resource
        isTerminalState = \
            np.all(np.take(self.state, 1, axis=1) < 0.01) or \
            np.all(np.take(self.state, 2, axis=1) < 0.01) or \
            np.all(np.take(self.state, 3, axis=1) < 0.01)


        return self.state, reward, isTerminalState, info


    def render(self):
        pass


    def reset(self):
        # An [n, 4] array representing the value, CPU, Memory and Storage capacities
        # of each of our bins, with an extra bin at the end representing the next VM.
        self.state = np.column_stack((
            np.full(self.num_bins + 1, np.float32(0)), # Value
            np.full(self.num_bins + 1, np.float32(self.capacity)), # CPU
            np.full(self.num_bins + 1, np.float32(self.capacity)), # Memory
            np.full(self.num_bins + 1, np.float32(self.capacity)) # Storage
            ))

        self.getNewItem()

        return self.state

    
    def getNewItem(self):
        index = np.random.randint(low=0, high=len(self.available_vms) - 1)
        next_item = self.available_vms[index][:-1] # We ignore the final, popularity, value for now
        self.state[self.num_bins] = next_item