import numpy as np

from actor import Actor
from mcts import generate_children, TreeNode
from neuralnet import NeuralNet


class Tournament:
    def __init__(self, nr_topp_games, checkpoint_path, M, episodes_per_game, game, topp_exploration_rate=0.00):
        self.NR_TOPP_GAMES = nr_topp_games
        self.CHECKPOINT_PATH = checkpoint_path
        self.M = M
        self.EPISODES_PER_GAME = episodes_per_game
        self.actors = []
        self.game = game
        self.topp_exploration_rate = topp_exploration_rate

    def load_actors(self):
        for i in range(self.M + 1):
            number = i * self.EPISODES_PER_GAME
            model = NeuralNet(checkpoint_path=self.CHECKPOINT_PATH,
                              episodes_per_game=self.EPISODES_PER_GAME,
                              label=str(number))
            model.load(number)
            self.actors.append(Actor(model, exploration_rate=self.topp_exploration_rate,
                                     exploration_rate_decay_fact=1, label=str(number)))
            print(f'\n\n{model.label}')

    def play_tournament(self):
        combinations = [(a, b) for idx, a in enumerate(self.actors) for b in self.actors[idx + 1:]]
        labels = [(pair[0].label, pair[1].label) for pair in combinations]
        scores = {}

        print(f'Combinations: {labels}')

        for pair in combinations:
            actor1 = pair[0]
            actor2 = pair[1]
            # actor1 = self.models[-1]
            # actor2 = self.models[0]
            if actor1.label not in scores.keys(): scores[actor1.label] = 0
            if actor2.label not in scores.keys(): scores[actor2.label] = 0
            print(f'Actor1: {actor1.label}')
            print(f'Actor2: {actor2.label}')

            first_player = actor1
            for game_i in range(self.NR_TOPP_GAMES):
                print(f'\n\n GAME {game_i} \t {actor1.label} vs {actor2.label}\n')
                game = self.game.create_copy()
                next_player = first_player
                while not game.is_game_over():
                    # game.render()
                    # print(f'Model to move: {next_player.label}')
                    node = TreeNode(game.state)
                    generate_children(node, game)
                    state = np.concatenate(
                        (np.ravel(game.state['board_state']), np.array([game.state['pid']], dtype='float')))

                    state = state.reshape(1, -1)
                    action, action_index = next_player.pick_action(state, node)
                    next_player = actor2 if next_player == actor1 else actor1
                    game.make_move(action)
                # game.render()
                winner = actor2.label if next_player == actor1 else actor1.label
                print(f'WINNER: {winner}')
                scores[winner] += 1
                first_player = actor2 if first_player == actor1 else actor1

        print('\n\n --- Tournament of Progressive Policies (TOPP) --- \n')
        print('RESULTS:')
        [print(f'Actor{label}: \t{scores[label]}') for label in scores.keys()]
