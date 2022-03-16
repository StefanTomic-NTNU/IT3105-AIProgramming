from simworld.nim import Nim


class TreeNode:
    def __init__(self, state):
        self.parent = None
        self.children = ()
        self.state = state
        self.score = 0
        self.N = 0

    def generate_children(self):
        pass


class MCTS:
    def __init__(self, number_actual_games, number_search_games):
        self.number_actual_games = number_actual_games
        self.number_search_games = number_search_games
        self.board = Nim(50, 10)

    def run(self):
        for g_a in range(self.number_actual_games):
            board_a = Nim(50, 10)
            state = self.board.get_state()
            root = TreeNode(state)
            while not board_a.is_game_over():
                board_mc = board_a.create_copy()
                for g_s in range(self.number_search_games):
                    actions = board_mc.get_legal_actions()
                    for action in actions:

