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
    nn = NeuralNet(lrate=config['learning_rate'], nn_dims=tuple(config['nn_dims']),
                   hidden_act_func=config['hidden_act_func'], optimizer=config['optimizer'],
                   episodes_per_game=episodes_per_game, checkpoint_path=config['checkpoint_path'])
    mcts = MCTS(episodes, config['nr_search_games'], nim, config['nn_dims'][0], nn,
                exploration_rate=config['init_exploration_rate'],
                exploration_rate_decay_fact=config['exploration_rate_decay_fact'])

    rl_system = ReinforcementLearningSystem()
    topp = Tournament()

    # hex.cells[0, 0].piece = (1, 0)
    # hex.cells[2, 2].piece = (0, 1)
    # hex.cells[4, 0].piece = (1, 0)
    # hex.render()
    # hex.state = {'board_state': hex.simplify_state(),
    #               'pid': 1}
    # print('state:')
    # print(hex.state['board_state'])
    # print(f'{len(hex.state["board_state"])} \n\n')
    # actions = hex.get_legal_actions()
    # print(actions)
    # print(f'{len(actions)} \n\n')

    hex.make_move(0)
    hex.make_move(5)

    hex.make_move(1)
    hex.make_move(6)

    hex.make_move(2)
    hex.make_move(7)

    hex.make_move(3)
    hex.make_move(8)

    hex.make_move(3)
    hex.make_move(2)

    hex.make_move(7)
    hex.make_move(4)

    hex.make_move(11)
    hex.make_move(10)

    hex.make_move(15)
    hex.make_move(8)
    hex.render()
    print(hex.is_game_over())

    # mcts.run()
