from mcts import MCTS, TreeNode
from simworld.nim import Nim
import numpy as np

if __name__ == '__main__':
    tree_search = MCTS(200, 500, Nim(10, 2), 2)
    tree_search.run()
    state = {
        'board_state': 2,
        'pid': 1
    }

    # node = TreeNode(state)
    # tree_search.generate_children(node)
    # nn_input = np.array([node.state['board_state'], node.state['pid']])
    # # input_tensor = tf.convert_to_tensor(nn_input, dtype=tf.int32)
    # nn_input = nn_input.reshape(1, -1)
    # action, action_index = tree_search.pick_action(nn_input, node)
    #
    # tree_search.model()

