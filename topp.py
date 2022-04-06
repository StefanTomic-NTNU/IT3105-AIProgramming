import numpy as np

from mcts import TreeNode
from neuralnet import NeuralNet


class Tournament:
    def __init__(self, nr_topp_games, checkpoint_path, M, episodes_per_game, game,
                 lrate=0.01, optimizer='SGD', loss='categorical_crossentropy', in_shape=(2,),
                 nn_dims=(1024, 512, 32, 1), hidden_act_func='relu'
                 ):
        self.NR_TOPP_GAMES = nr_topp_games
        self.CHECKPOINT_PATH = checkpoint_path
        self.M = M
        self.EPISODES_PER_GAME = episodes_per_game
        self.models = []
        self.game = game
        self.in_shape = in_shape
        self.nn_dims = nn_dims

    def load_models(self):
        for i in range(self.M + 1):
            number = i * self.EPISODES_PER_GAME
            model = NeuralNet(checkpoint_path=self.CHECKPOINT_PATH,
                              episodes_per_game=self.EPISODES_PER_GAME,
                              label=str(number),
                              in_shape=self.in_shape,
                              nn_dims=self.nn_dims)
            model.load_weights(number)
            self.models.append(model)
            print(f'\n\n{model.label}')
            print(model.model.layers[0].get_weights()[1])


    def play_tournament(self):
        combinations = [(a, b) for idx, a in enumerate(self.models) for b in self.models[idx + 1:]]
        labels = [(pair[0].label, pair[1].label) for pair in combinations]
        scores = {}

        print(f'Combinations: {labels}')

        for pair in combinations:
            model1 = pair[0]
            model2 = pair[1]
            # model1 = self.models[-1]
            # model2 = self.models[0]
            if model1.label not in scores.keys(): scores[model1.label] = 0
            if model2.label not in scores.keys(): scores[model2.label] = 0
            print(f'Model1: {model1.label}')
            print(f'Model2: {model2.label}')

            first_player = model1
            for game_i in range(self.NR_TOPP_GAMES):
                print(f'\n\n GAME {game_i} \t {model1.label} vs {model2.label}\n')
                game = self.game.create_copy()
                next_player = first_player
                while not game.is_game_over():
                    # game.render()
                    print(f'Model to move: {next_player.label}')
                    node = TreeNode(game.state)
                    self._generate_children(node)
                    state = np.concatenate(
                        (np.ravel(game.state['board_state']), np.array([game.state['pid']], dtype='float')))

                    state = state.reshape(1, -1)
                    action, action_index = self._get_greedy_action(state, node, next_player)
                    next_player = model2 if next_player == model1 else model1
                    game.make_move(action)
                game.render()
                winner = model2.label if next_player == model1 else model1.label
                print(f'WINNER: {winner}')
                scores[winner] += 1
                first_player = model2 if first_player == model1 else model1

        print('\n\n --- Tournament of Progressive Policies (TOPP) --- \n')
        print('RESULTS:')
        [print(f'Model{label}: \t{scores[label]}') for label in scores.keys()]

    def _generate_children(self, tree_node: TreeNode):
        if len(tree_node.children) == 0 and tree_node.state:
            edges, states, illegal_edges, illegal_states = self.game.generate_children_(tree_node.state)
            children = [TreeNode(child) for child in states]
            illegal_children = [TreeNode(illegal_child) for illegal_child in illegal_states]
            for i in range(len(children)):
                self._add_child(tree_node, children[i], edges[i])
            for j in range(len(illegal_children)):
                illegal_children[j].is_illegal = True
                illegal_children[j].N = 0
                self._add_child(tree_node, illegal_children[j], illegal_edges[j])

    def _add_child(self, parent, child, edge):
        if child not in parent.children:
            parent.children.append(child)
            parent.edges.append(edge)
            if not child.is_illegal:
                parent.N_a.append(1)
            else:
                parent.N_a.append(0)
            parent.score_a.append(0)
            parent.Q_a.append(0)
            child.parent = parent

    def _get_greedy_action(self, state, node: TreeNode, model, verbose=False):
        action_tensor = model.predict(state)
        action_dist = action_tensor.numpy()[0]
        action_index = np.argmax(action_dist)
        while node.children[action_index].is_illegal or action_index >= len(node.edges):
            action_dist[action_index] = 0
            action_dist = _normalize(action_dist)
            action_index = np.argmax(action_dist)
        if verbose: print(f'Action dist: {action_dist}')
        action = node.edges[action_index]
        return action, action_index

def _normalize(arr: np.array):
    return arr / np.sum(arr)