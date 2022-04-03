import json

from mcts import MCTS, TreeNode
from neuralnet import NeuralNet
from rl_system import ReinforcementLearningSystem
from simworld.nim import Nim
import numpy as np

from topp import Tournament


def read_config():
    with open('config.json', 'r') as f:
        config_ = json.load(f)
    return config_


if __name__ == '__main__':
    config = read_config()

    game = Nim(config['nim_pieces'], config['nim_k'])
    nn = NeuralNet(lrate=config['learning_rate'], nn_dims=tuple(config['nn_dims']),
                   hidden_act_func=config['hidden_act_func'], optimizer=config['optimizer'],
                   M=config['M'])
    mcts = MCTS(config['nr_episodes'], config['nr_search_games'], game, config['nn_dims'][0], nn)

    rl_system = ReinforcementLearningSystem()
    topp = Tournament()

    mcts.run()
