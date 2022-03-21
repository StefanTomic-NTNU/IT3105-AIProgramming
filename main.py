from mcts import MCTS
from simworld.nim import Nim

if __name__ == '__main__':
    tree_search = MCTS(50, 20, Nim(20, 3))
    tree_search.run()
