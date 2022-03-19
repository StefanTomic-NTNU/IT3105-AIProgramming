from simworld.nim import Nim


class TreeNode:
    def __init__(self, state: dict):
        self.parent = None
        self.edges = []
        self.children = []
        self.state = state
        self.score = 0
        self.N = 0
        self.Q = 0


class MCTS:
    def __init__(self, number_actual_games, number_search_games, game: Nim):
        self.number_actual_games = number_actual_games
        self.number_search_games = number_search_games
        self.game = game

    def run(self):
        for g_a in range(self.number_actual_games):
            board_a = self.game
            init_state = board_a.state
            root = TreeNode(init_state)
            while not board_a.is_game_over():
                board_mc = self.game
                board_mc.state = root.state

                for g_s in range(self.number_search_games):
                    node = root
                    while len(node.children) != 0:
                        traversing_node = node.children[0]  # TODO: Replace tree policy
                        node = traversing_node
                        board_mc.state = traversing_node.state
                    self.generate_children(node)
                    # Rollout:
                    rollout_nodes = []
                    # origin = TreeNode(node.state)
                    while not board_mc.is_game_over():
                        self.generate_children(node)
                        # TODO: Select action using ANET
                        board_mc.make_move(node.edges[0])
                        node = TreeNode(board_mc.state)
                        rollout_nodes.append(node)
                    if board_mc.state['pid'] == 1:
                        eval = -1
                    else:
                        eval = 1
                    node = rollout_nodes[0]
                    parent = node.parent
                    while parent:
                        node.score += eval
                        node.N += 1
                        node.Q = node.score / node.N
                        parent = node.parent
                # TODO: Get D   ;)
                action = root.edges[0]  # TODO: Choose action based on D
                board_a.make_move(action)
                root = root.children[0]
                root.parent = None

    def generate_children(self, tree_node: TreeNode):
        edges, states = self.game.generate_children_(tree_node.state)
        children = [TreeNode(child) for child in states]
        for i in range(len(children)):
            self.add_child(tree_node, children[i], edges[i])

    def add_child(self, parent, child, edge):
        if child not in parent.children:
            parent.children.append(child)
            parent.edges.append(edge)
            child.parent = parent
