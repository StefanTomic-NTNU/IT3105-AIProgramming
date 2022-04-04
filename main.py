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

    hex = Hex(config['hex_size'], init_player=2)
    nim = Nim(config['nim_pieces'], config['nim_k'])

    nn = NeuralNet(lrate=config['learning_rate'], in_shape=(config['in_shape'], ),
                   nn_dims=tuple(config['nn_dims']),
                   hidden_act_func=config['hidden_act_func'], optimizer=config['optimizer'],
                   episodes_per_game=episodes_per_game, checkpoint_path=config['checkpoint_path'])
    mcts = MCTS(episodes, config['nr_search_games'], hex, config['nn_dims'][0], nn,
                exploration_rate=config['init_exploration_rate'],
                exploration_rate_decay_fact=config['exploration_rate_decay_fact'])

    rl_system = ReinforcementLearningSystem()
    topp = Tournament()

    mcts.run()
