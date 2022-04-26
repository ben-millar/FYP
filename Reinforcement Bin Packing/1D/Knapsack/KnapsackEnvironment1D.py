from gym.spaces import Discrete, Box
import numpy as np


class KnapsackPacking(object):
    metadata = {}

    CAPACITIES = 0
    VALUES = 1

    def __init__(self, num_knapsacks, capacity, min_item_size=1, max_item_size=10, min_item_value=1, max_item_value=100) -> None:
        self.capacity = capacity # Max capacity of a bin
        self.num_knapsacks = num_knapsacks
        self.min_item_size = min_item_size
        self.max_item_size = max_item_size
        self.min_item_value = min_item_value
        self.max_item_value = max_item_value

        # Initialise environment
        self.state = self.reset()

        min = np.vstack((
            np.full((self.num_knapsacks + 1), 0),
            np.full((self.num_knapsacks + 1), 0)
        ))

        max = np.vstack((
            np.full((self.num_knapsacks + 1), self.capacity),
            np.full((self.num_knapsacks + 1), max_item_value * (capacity/min_item_size))
        ))

        # Our observation space is each possible value of a bin
        self.observation_space = Box(
            low=min,
            high=max,
            shape=(2,self.num_knapsacks+1),
            dtype=np.int
        )
        
        # Our actions will index into our array of bins, or reject the item
        self.action_space = Discrete(len(self.state[0]))
        
        self.reward_range = (-max_item_size, (max_item_value*max_item_value*max_item_value)/100)

        self.logs = { 'placed':0, 'misplaced':0, 'discarded':0 }


    def step(self, action):
        item_size = self.state[self.CAPACITIES][self.num_knapsacks]
        item_value = self.state[self.VALUES][self.num_knapsacks]

        if action == len(self.state[self.CAPACITIES]) - 1:
            # Discard item without trying to place it
            reward = -10
            self.getNewItem()
            self.logs['discarded'] = self.logs['discarded'] + 1
        elif self.state[self.CAPACITIES][action] < item_size:
            # Attempted to place item in a bin that was too small
            reward = -10
            self.logs['misplaced'] = self.logs['misplaced'] + 1
        else:
            # Successfully placed item in a bin
            self.state[self.CAPACITIES][action] -= item_size
            self.state[self.VALUES][action] += item_value

            reward = item_value

            self.getNewItem()

            self.logs['placed'] = self.logs['placed'] + 1

        # Returns true if all of our bins are at capacity
        isTerminalState = np.all((self.min_item_size >= self.state[self.CAPACITIES]))

        # Empty placeholder debug info dict
        info = {}

        return self.state, reward, isTerminalState, info


    def getNewItem(self):
        self.state[self.CAPACITIES][self.num_knapsacks] = np.random.randint(low=self.min_item_size, high=self.max_item_size)
        self.state[self.VALUES][self.num_knapsacks] = np.random.randint(low=self.min_item_value, high=self.max_item_value)


    def render(self):
        pass


    def reset(self):
        # np array representing capacities of each knapsack,
        # with an extra item at the end representing the next to be placed
        capacities = np.full((self.num_knapsacks + 1), self.capacity)

        # np array representing values of each knapsack,
        # with an extra item at the end representing the next to be placed
        values = np.full((self.num_knapsacks + 1), 0)

        # Stitch our [n] arrays into one [2, n] array
        self.state = np.vstack((capacities, values))

        self.getNewItem()

        return self.state
