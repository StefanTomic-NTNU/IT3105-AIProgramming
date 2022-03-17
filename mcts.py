from simworld.nim import Nim


class TreeNode:
    def __init__(self, sim: Nim):
        self.parent = None
        self.children = set()
        self.sim = sim
        self.score = 0
        self.N = 0

    def add_child(self, child):
        self.children.add(child)
        child.parent = self

    def generate_children(self):
        children = self.sim.generate_children()
        for child in children:
            self.add_child(TreeNode(child))



class MCTS:
    def __init__(self, number_actual_games, number_search_games):
        self.number_actual_games = number_actual_games
        self.number_search_games = number_search_games
        self.board = Nim(50, 10)

    def run(self):
        for g_a in range(self.number_actual_games):
            board_a = Nim(50, 10)
            # state = self.board.get_state()
            root = TreeNode(board_a)
            while not board_a.is_game_over():
                board_mc = board_a.create_copy()
                board_mc_node = TreeNode(board_mc)
                for g_s in range(self.number_search_games):
                    board_mc_node.generate_children()

