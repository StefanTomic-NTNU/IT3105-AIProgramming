import random

import numpy as np

from mcts import TreeNode
from neuralnet import NeuralNet


class Actor:
    def __init__(self, nn: NeuralNet, exploration_rate=0.50, exploration_rate_decay_fact=0.99, label=''):
        self.nn = nn
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay_fact = exploration_rate_decay_fact
        self.label = label

    def fit(self, x, y, callbacks=None):
        self.nn.fit(x, y, callbacks=callbacks)

    def pick_action(self, state, node, verbose=False):
        if random.uniform(0, 1) < 1 - self.exploration_rate:
            return self.get_greedy_action(state, node, verbose=verbose)
        else:
            illegal_move = True
            index = random.randrange(len(node.edges))
            while illegal_move:
                index = random.randrange(len(node.edges))
                illegal_move = node.children[index].is_illegal
            return node.edges[index], index

    def get_greedy_action(self, state, node: TreeNode, verbose=False):
        action_tensor = self.nn.predict(state)
        action_dist = action_tensor[0]
        action_index = np.argmax(action_dist)
        while node.children[action_index].is_illegal or action_index >= len(node.edges):
            action_dist[action_index] = 0
            action_dist = normalize(action_dist)
            action_index = np.argmax(action_dist)
        if verbose: print(f'Action dist: {action_dist}')
        action = node.edges[action_index]
        return action, action_index

    def decay_exploration_rate(self):
        self.exploration_rate *= self.exploration_rate_decay_fact


def normalize(arr: np.array):
    return arr/np.sum(arr)
