from mcts import MCTS
from simworld.nim import Nim

if __name__ == '__main__':
    tree_search = MCTS(2, 10, Nim(20, 3))
    tree_search.run()
