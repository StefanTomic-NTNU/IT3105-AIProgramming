import copy
import math
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras as KER
import matplotlib.pyplot as plt

from simworld.nim import Nim


class TreeNode:
    def __init__(self, state: dict):
        self.is_illegal = False
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

    def is_at_end(self):
        if not self.children:
            return False
        no_legal_moves = True
        for child in self.children:
            if not child.is_illegal:
                no_legal_moves = False
        return no_legal_moves


class MCTS:
    def __init__(self, number_actual_games, number_search_games, game, nr_actions, nn,
                 exploration_rate=0.50, exploration_rate_decay_fact=0.98):
        self.number_actual_games = number_actual_games
        self.number_search_games = number_search_games
        self.game = game
        self.nr_actions = nr_actions
        self.model = nn
        self.replay_buffer = []
        self.prob_disc_dict = {}
        self.exploration_rate = exploration_rate
        self.exploration_rate_decay_fact = exploration_rate_decay_fact

        # Include the epoch in the file name (uses `str.format`)
        self.checkpoint_path = "models/cp-{epoch:04d}.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        # Create a callback that saves the model's weights every 5 epochs
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq=20)

    def run(self):
        for g_a in range(self.number_actual_games):
            board_a = self.game.create_copy()
            root = TreeNode(copy.copy(board_a.state))
            while not board_a.is_game_over():
                board_mc = self.game.create_copy()
                board_mc.set_state(root.state)

                for g_s in range(self.number_search_games):
                    board_mc = self.game.create_copy()
                    board_mc.set_state(root.state)

                    # TREE POLICY
                    node = root
                    while node.children and not node.is_at_end():
                        chosen_node = self.tree_policy(node)
                        if chosen_node is None: break
                        if node.children[chosen_node].state is None: break
                        board_mc.make_move(node.edges[chosen_node])
                        node = node.children[chosen_node]

                    self.generate_children(node)    # Blue nodes

                    # ROLLOUT
                    grey_node = node
                    while node.state and not board_mc.is_game_over():
                        self.generate_children(node)

                        nn_input = np.concatenate((np.ravel(node.state['board_state']), np.array([node.state['pid']], dtype='float')))
                        # nn_input = np.array([node.state['board_state'], node.state['pid']])
                        nn_input = nn_input.reshape(1, -1)
                        action, action_index = self.pick_action(nn_input, node)
                        if not board_mc.is_game_over():
                            board_mc.make_move(action)
                            node = node.children[action_index]

                    # BACKPROPAGATION
                    evaluation = -1 if board_mc.state['pid'] == 1 else 1
                    parent = node.parent
                    while parent:
                        node.score += evaluation
                        node.N += 1
                        node.Q = node.score / node.N
                        edge_index = node.parent.children.index(node)
                        node.parent.score_a[edge_index] += evaluation
                        node.parent.N_a[edge_index] += 1
                        node.parent.Q_a[edge_index] = node.parent.score_a[edge_index] / node.parent.N_a[edge_index]
                        node = node.parent
                        parent = node.parent

                    # Cleanup children:
                    if not grey_node.is_at_end():
                        for child in grey_node.children:
                            child.score_a = []
                            child.N_a = []
                            child.Q_a = []
                            child.edges = []
                            child.children = []

                # root_state = np.array([root.state['board_state'], root.state['pid']])
                root_state = np.concatenate((np.ravel(root.state['board_state']), np.array([root.state['pid']], dtype='float')))
                root_state = root_state.reshape(1, -1)
                D = copy.copy(normalize(np.array([root.N_a])))
                case = (root_state, D)
                self.replay_buffer.append(case)
                # self.prob_disc_dict[(root.state['board_state'], root.state['pid'])] = D
                action, action_index = self.pick_action(case[0], root)
                board_a.make_move(action)
                root = root.children[action_index]
                root.parent = None
            batch_size = len(self.replay_buffer)
            number_from_batch = random.randrange(math.floor(batch_size/5), batch_size)
            if number_from_batch == 0: number_from_batch = 1
            subbatch = random.sample(self.replay_buffer, number_from_batch)

            ex_batch_x = subbatch[0][0][0]
            ex_batch_y = subbatch[0][1][0]

            # op_batch_x = np.zeros((len(self.prob_disc_dict), len(ex_batch_x)))
            # op_batch_y = np.zeros((len(self.prob_disc_dict), len(ex_batch_y)))

            # i = 0
            # for key in self.prob_disc_dict.keys():
            #     op_batch_x[i, :] = np.array(list(key))
            #     op_batch_y[i, :] = np.array(self.prob_disc_dict[key])
            #     i += 1

            # TODO: Mål tid, effektiviser?

            batch_x = np.zeros((number_from_batch, len(ex_batch_x)))
            for i in range(number_from_batch):
                batch_x[i, :] = subbatch[i][0]
            batch_y = np.zeros((number_from_batch, len(ex_batch_y)))
            for i in range(number_from_batch):
                batch_y[i, :] = subbatch[i][1]
            # for i in range(len(self.replay_buffer)):
            #     print(self.replay_buffer[i])

            self.model.fit(x=batch_x, y=batch_y)
            self.exploration_rate *= self.exploration_rate_decay_fact
#        self.model.save(self.checkpoint_path.format(epoch=1337))

        # "OPTIMAL" GAME
        self.exploration_rate = 0
        # self.model.load_weights(200)
        for init_player in (1, 2):
            final_game = self.game.create_copy()
            self.game.state['pid'] = init_player
            while not final_game.is_game_over():
                print(f'Final game pieces: {final_game.state["board_state"]} \t Player: {final_game.state["pid"]}')
                node = TreeNode(final_game.state)
                self.generate_children(node)
                state = np.array([final_game.state['board_state'], final_game.state['pid']])
                state = state.reshape(1, -1)
                action, action_index = self.pick_action(state, node)
                print(f'Action {action}')
                final_game.make_move(action)
            print(f'Final game pieces: {final_game.state["board_state"]} \t Player: {final_game.state["pid"]}')
            winner = 3 - final_game.state['pid']
            print(f'Winner is player {winner}\n\n')

            print(f'Model predictions: ')
            for i in range(1, 11):
                node = TreeNode({'board_state': i, 'pid': init_player})
                print(f'State: [{i}, {init_player}]')
                self.generate_children(node)
                state = np.array([i, init_player])
                state = state.reshape(1, -1)
                action, action_index = self.pick_action(state, node, verbose=True)
                if (i, init_player) in self.prob_disc_dict:
                    print(f'Training case extreme: {self.prob_disc_dict[(i, init_player)]}\n')

    def generate_children(self, tree_node: TreeNode):
        if len(tree_node.children) == 0 and tree_node.state:
            edges, states, illegal_edges, illegal_states = self.game.generate_children_(tree_node.state)
            children = [TreeNode(child) for child in states]
            illegal_children = [TreeNode(illegal_child) for illegal_child in illegal_states]
            for i in range(len(children)):
                self.add_child(tree_node, children[i], edges[i])
            for j in range(len(illegal_children)):
                illegal_children[j].is_illegal = True
                illegal_children[j].N = 0
                self.add_child(tree_node, illegal_children[j], illegal_edges[j])

    def add_child(self, parent, child, edge):
        if child not in parent.children:
            parent.children.append(child)
            parent.edges.append(edge)
            if not child.is_illegal:
                parent.N_a.append(1)    # TODO: to avoid 0 log
            else:
                parent.N_a.append(0)
            parent.score_a.append(0)
            parent.Q_a.append(0)
            child.parent = parent

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
        action_tensor = self.model.predict(state)
        action_dist = action_tensor.numpy()[0]
        action_index = np.argmax(action_dist)
        while node.children[action_index].is_illegal or action_index >= len(node.edges):
            action_dist[action_index] = 0
            action_dist = normalize(action_dist)
            action_index = np.argmax(action_dist)
        if verbose: print(f'Action dist: {action_dist}')
        action = node.edges[action_index]
        return action, action_index

    def tree_policy(self, node: TreeNode):
        u = [1*np.sqrt(np.log(node.N)/(1 + N_sa)) for N_sa in node.N_a]

        if node.state['pid'] == 1:
            combined = np.add(node.Q_a, u)
            policy = np.argmax(combined)
        else:
            combined = np.subtract(node.Q_a, u)
            policy = np.argmin(combined)

        while node.children[policy].state is None:
            if node.state['pid'] == 1:
                combined[policy] = -100000
                policy = np.argmax(combined)
                if np.sum(combined) == -100000 * self.nr_actions:
                    return None
            else:
                combined[policy] = 100000
                policy = np.argmin(combined)
                if np.sum(combined) == 100000 * self.nr_actions:
                    return None
        return policy


def normalize(arr: np.array):
    return arr/np.sum(arr)
