import json

from mcts import MCTS, TreeNode
from neuralnet import NeuralNet
from rl_system import ReinforcementLearningSystem
from simworld.hex_board import Hex
from simworld.nim import Nim
import numpy as np

from topp import Tournament


def read_config():
    with open('config.json', 'r') as f:
        config_ = json.load(f)
    return config_


if __name__ == '__main__':
    config = read_config()

    episodes = config['nr_episodes']
    M = config['M']
    episodes_per_game = round(episodes / M)

    nn_dims = config['nn_dims']
    game_name = config['game']
    if game_name == 'hex' or game_name == 'Hex':
        size = config['hex_size']
        game = Hex(size)
        checkpoint_path = 'models/hex' + \
                          str(config['hex_size']) + '/' + \
                          config['checkpoint_path']
        nn_dims[-1] = size ** 2
        in_shape = (2 * (size ** 2) + 1, )
    elif game_name == 'nim' or game_name == 'Nim':
        k = config['nim_k']
        game = Nim(config['nim_pieces'], k)
        checkpoint_path = 'models/nim' + \
                          str(config['nim_k']) + '-' + str(config['nim_k']) + '/' + \
                          config['checkpoint_path']
        nn_dims[-1] = k
        in_shape = (2, )
    else:
        raise Exception

    nn = NeuralNet(lrate=config['learning_rate'], in_shape=in_shape,
                   nn_dims=tuple(nn_dims),
                   hidden_act_func=config['hidden_act_func'], optimizer=config['optimizer'],
                   episodes_per_game=episodes_per_game, checkpoint_path=checkpoint_path)
    mcts = MCTS(episodes, config['nr_search_games'], game, nn_dims[-1], nn,
                exploration_rate=config['init_exploration_rate'],
                exploration_rate_decay_fact=config['exploration_rate_decay_fact'])

    rl_system = ReinforcementLearningSystem()
    topp = Tournament()

    mcts.run()
