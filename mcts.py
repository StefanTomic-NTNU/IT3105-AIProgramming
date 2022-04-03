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
    def __init__(self, number_actual_games, number_search_games, game: Nim, nr_actions):
        self.number_actual_games = number_actual_games
        self.number_search_games = number_search_games
        self.game = game
        self.nr_actions = nr_actions
        self.model = self.gennet(num_classes=nr_actions)
        self.replay_buffer = []
        self.random_prob = 0.50
        self.random_prob_decay_rate = 0.95

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
        # TODO: Bedre kontroll på hvilke noder som eksisterer (følger av root) + Eval dersom tree policy finner en end_state
        for g_a in range(self.number_actual_games):
            board_a = self.game.create_copy()
            root = TreeNode(copy.copy(board_a.state))
            while not board_a.is_game_over():
                board_mc = self.game.create_copy()
                board_mc.state = copy.copy(root.state)

                for g_s in range(self.number_search_games):

                    # TREE POLICY
                    node = root
                    while node.children and not node.is_at_end():
                        chosen_node = self.tree_policy(node)
                        if chosen_node is None: break
                        if node.children[chosen_node].state is None: break
                        node = node.children[chosen_node]
                        board_mc.state = copy.copy(node.state)

                    self.generate_children(node)    # Blue nodes

                    # ROLLOUT
                    grey_node = node
                    while node.state and not board_mc.is_game_over():
                        self.generate_children(node)
                        nn_input = np.array([node.state['board_state'], node.state['pid']])
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

                root_state = np.array([root.state['board_state'], root.state['pid']])
                root_state = root_state.reshape(1, -1)
                D = normalize(np.array([root.N_a]))
                case = (root_state, D)
                self.replay_buffer.append(case)
                action, action_index = self.pick_action(case[0], root)
                board_a.make_move(action)
                root = root.children[action_index]
                root.parent = None
            batch_size = len(self.replay_buffer)
            number_from_batch = random.randrange(math.ceil(batch_size/5), batch_size)
            subbatch = random.sample(self.replay_buffer, number_from_batch)

            # TODO: Train using vector of vectors:
            print(f'RBUF: {self.replay_buffer}')
            print(f'Number for subbatch: {number_from_batch}')
            print(f'Subbatch: {subbatch}')

            ex_batch_x = subbatch[0][0][0]
            ex_batch_y = subbatch[0][1][0]
            batch_x = np.zeros((number_from_batch, len(ex_batch_x)))
            for i in range(number_from_batch):
                batch_x[i, :] = subbatch[i][0]
            batch_y = np.zeros((number_from_batch, len(ex_batch_y)))
            for i in range(number_from_batch):
                batch_y[i, :] = subbatch[i][1]
            for i in range(len(self.replay_buffer)):
                print(self.replay_buffer[i])

            self.model.fit(x=batch_x, y=batch_y, callbacks=[self.cp_callback])
            self.random_prob *= self.random_prob_decay_rate
        self.model.save_weights(self.checkpoint_path.format(epoch=1337))

        # "OPTIMAL" GAME
        self.random_prob = 0
        for init_player in (1, 2):
            final_game = Nim(10, 2, init_player=init_player)
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

    def pick_action(self, state, node):
        if random.uniform(0, 1) < 1 - self.random_prob:
            return self.get_greedy_action(state, node)
        else:
            illegal_move = True
            index = random.randrange(len(node.edges))
            while illegal_move:
                index = random.randrange(len(node.edges))
                illegal_move = node.children[index].is_illegal
            return node.edges[index], index

    def get_greedy_action(self, state, node):
        action_tensor = self.model(state)
        action_dist = action_tensor.numpy()[0]
        action_index = np.argmax(action_dist)
        while node.children[action_index].is_illegal:
            action_dist[action_index] = 0
            action_dist = normalize(action_dist)
            action_index = np.argmax(action_dist)
        while action_index >= len(node.edges):
            action_dist[action_index] = 0
            action_dist = normalize(action_dist)
            action_index = np.argmax(action_dist)
        action = node.edges[action_index]
        return action, action_index

    def tree_policy(self, node: TreeNode):
        u = [5*np.sqrt(np.log(node.N)/(1 + N_sa)) for N_sa in node.N_a]
        combined = np.add(u, node.Q_a)
        policy = np.argmax(combined)
        while node.children[policy].state is None:
            combined[policy] = -100000
            policy = np.argmax(combined)
            if np.sum(combined) == -100000 * self.nr_actions:
                return None
        return policy

    def gennet(self, num_classes=2, lrate=0.0001, optimizer='SGD', loss='categorical_crossentropy', in_shape=(2,)):
        optimizer = eval('KER.optimizers.' + optimizer)
        loss = eval('KER.losses.' + loss) if type(loss) == str else loss

        model = KER.Sequential()
        model.add(KER.layers.Dense(20, input_shape=in_shape, activation='relu', name='input_layer'))
        model.add(KER.layers.Dense(512, activation='relu', name='middle_layer1'))
        model.add(KER.layers.Dense(256, activation='relu', name='middle_layer2'))
        model.add(KER.layers.Dense(num_classes, activation='softmax', name='output_layer'))

        model.compile(optimizer=optimizer(learning_rate=lrate), loss=loss, metrics=[KER.metrics.categorical_accuracy])
        return model


def normalize(arr: np.array):
    return arr/np.sum(arr)
