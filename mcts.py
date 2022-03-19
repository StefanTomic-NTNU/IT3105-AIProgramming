import copy
import random
import numpy as np

from simworld.nim import Nim


class TreeNode:
    def __init__(self, state: dict):
        self.parent = None
        self.score_a = []
        self.N_a = []
        self.Q_a = []
        self.edges = []
        self.children = []
        self.state = copy.copy(state)
        self.score = 0
        self.N = 1
        self.Q = 0


class MCTS:
    def __init__(self, number_actual_games, number_search_games, game: Nim):
        self.number_actual_games = number_actual_games
        self.number_search_games = number_search_games
        self.game = game

    def run(self):
        for g_a in range(self.number_actual_games):
            board_a = self.game
            root = TreeNode(copy.copy(board_a.state))
            while not board_a.is_game_over():
                board_mc = self.game
                board_mc.state = copy.copy(root.state)

                for g_s in range(self.number_search_games):
                    node = root
                    while len(node.children) != 0:
                        traversing_node = node.children[self.tree_policy(node)]  # TODO: Replace tree policy
                        node = traversing_node
                        board_mc.state = copy.copy(traversing_node.state)
                    self.generate_children(node)
                    # Rollout:
                    rollout_nodes = []
                    # origin = TreeNode(node.state)
                    while not board_mc.is_game_over():
                        self.generate_children(node)
                        # TODO: Select action using ANET
                        chosen_action = random.randrange(len(node.edges))
                        board_mc.make_move(node.edges[chosen_action])
                        node = node.children[chosen_action]
                        rollout_nodes.append(node)
                    if board_mc.state['pid'] == 1:
                        eval = -1
                    else:
                        eval = 1
                    node = rollout_nodes[0]
                    parent = node.parent
                    while parent:
                        # print('parent')
                        node.score += eval
                        node.N += 1
                        node.Q = node.score / node.N
                        edge_index = node.parent.children.index(node)
                        node.parent.score_a[edge_index] += eval
                        node.parent.N_a[edge_index] += 1
                        node.parent.Q_a[edge_index] = node.parent.score_a[edge_index] / node.parent.N_a[edge_index]
                        node = node.parent
                        parent = node.parent
                # TODO: Get D   ;)
                action = root.edges[0]  # TODO: Choose action based on D
                board_a.make_move(action)
                root = root.children[0]
                root.parent = None

    def generate_children(self, tree_node: TreeNode):
        if len(tree_node.children) == 0:
            edges, states = self.game.generate_children_(tree_node.state)
            children = [TreeNode(child) for child in states]
            for i in range(len(children)):
                self.add_child(tree_node, children[i], edges[i])

    def add_child(self, parent, child, edge):
        if child not in parent.children:
            parent.children.append(child)
            parent.edges.append(edge)
            parent.N_a.append(1)    # TODO: to avoid 0 log
            parent.score_a.append(0)
            parent.Q_a.append(0)
            child.parent = parent

    def tree_policy(self, node: TreeNode):
        u = [1*np.sqrt(np.log(node.N)/(1 + N_sa)) for N_sa in node.N_a]
        combined = np.add(u, node.Q_a)
        return np.argmax(combined)
