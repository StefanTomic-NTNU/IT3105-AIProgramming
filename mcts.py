import copy
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras as KER
import matplotlib.pyplot as plt

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
        self.model = self.gennet()
        self.replay_buffer = []

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
                        traversing_node = node.children[self.tree_policy(node)]  # TODO: Replace tree policy ?
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
                        nn_input = np.array([node.state['board_state'], node.state['pid']])
                        input_tensor = tf.convert_to_tensor(nn_input, dtype=tf.int32)
                        print(input_tensor)
                        action_tensor = self.model.call(input_tensor)
                        print(action_tensor)
                        chosen_action = tf.math.argmax(action_tensor, axis=1)
                        chosen_action = chosen_action.numpy()[0]
                        print(chosen_action)
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
                root_state = (root.state['board_state'], root.state['pid'])
                D = copy.copy(root.N_a)
                case = (root_state, D)
                self.replay_buffer.append(case)
                self.model.call(case)
                action = root.edges[0]  # TODO: Choose action based on D
                board_a.make_move(action)
                root = root.children[0]
                root.parent = None
            batch_size = len(self.replay_buffer)
            number_from_batch = random.randrange(round(batch_size/5), batch_size)
            subbatch = random.sample(self.replay_buffer, number_from_batch)
            for minibatch in subbatch:
                self.model.fit(x=minibatch[0], y=minibatch[1])

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

    def gennet(self, num_classes=3, lrate=0.01, optimizer='SGD', loss='categorical_crossentropy', in_shape=(1,)):
        optimizer = eval('KER.optimizers.' + optimizer)
        loss = eval('KER.losses.' + loss) if type(loss) == str else loss
        input_layer = KER.layers.Input(shape=in_shape, name='input_layer')
        x = input_layer
        x = KER.layers.Dense(30, activation='relu')(x)
        x = KER.layers.Dense(15, activation='relu')(x)
        output_layer = KER.layers.Dense(3, activation='softmax')(x)
        model = KER.models.Model(input_layer, output_layer)
        model = KER.Sequential()

        model.add(KER.layers.Dense(2, input_shape=(1,), activation='relu', name='input_layer'))
        model.add(KER.layers.Dense(12, activation='relu', name='middle_layer'))
        model.add(KER.layers.Dense(3, activation='softmax', name='output_layer'))

        model.compile(optimizer=optimizer(learning_rate=lrate), loss=loss, metrics=[KER.metrics.categorical_accuracy])
        return model

    def update_td_error(self, prev_state, new_state, reward, done):
        prev_state = tf.convert_to_tensor(tuple_to_np_array(prev_state))
        new_state = tf.convert_to_tensor(tuple_to_np_array(new_state))
        with tf.GradientTape() as tape:
            loss, td_error_tensor = get_loss(
                reward +
                self.model(new_state),
                self.model(prev_state)
            )
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # self.__td_error = tf.keras.backend.eval(td_error_tensor)[0][0]


@tf.function
def get_loss(true_target, predicted_target):
    td_error_tensor = true_target - predicted_target
    loss = td_error_tensor**2
    return loss, td_error_tensor


def tuple_to_np_array(tup): return np.array(np.asarray(tup).flatten().reshape(1, -1))