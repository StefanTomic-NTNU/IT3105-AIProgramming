class Nim:
    def __init__(self, n, k, init_player=1):
        if not 0 < init_player <= 2:
            raise Exception(f'Init player must be either 1 or 2, not {init_player}')
        self.state = {
            'board_state': n,
            'pid': init_player
                      }
        self.k = k  # max num pieces taken

    def make_move(self, action):
        if self.is_game_over():
            print('Game is over')
            return
        if action > self.k:
            raise Exception(f'Cannot take {action} pieces, when {self.k} is the max')
        if action > self.state['board_state']:
            self.state['board_state'] = 0
        else:
            self.state['board_state'] -= action
        self.state['pid'] = 3 - self.state['pid']

    def is_game_over(self):
        return self.state['board_state'] <= 0

    def get_legal_actions(self):
        return tuple(list(range(1, self.state['board_state']+1))) if self.k > self.state['board_state'] else tuple(list(range(1, self.k+1)))

    def render(self):
        if self.is_game_over():
            game_over = 'Game is over'
        else:
            game_over = 'Game is not over'
        print(f'Pieces left: {self.state["board_state"]} \t\t Player to move: {self.state["pid"]} \t\t {game_over} \t\t '
              f'Legal actions: {self.get_legal_actions()} ')

    def create_copy(self):
        return Nim(self.state['board_state'], self.k, init_player=self.state['pid'])

    def generate_children(self):
        edges = []
        children = []
        actions = self.get_legal_actions()
        for action in actions:
            dummy_game = self.create_copy()
            dummy_game.make_move(action)
            edges.append(action)
            children.append(dummy_game.state)
        return edges, children

    def step_(self, state, action):
        if self.is_game_over_(state):
            print('Game is over')
            return
        new_state = {
            'board_state': state['board_state'],
            'pid': state['pid']
                      }
        if action > self.k:
            raise Exception(f'Cannot take {action} pieces, when {self.k} is the max')
        if action > state['board_state']:
            new_state['board_state'] = 0
        else:
            new_state['board_state'] -= action
        new_state['pid'] = 3 - state['pid']
        return new_state

    def is_game_over_(self, state):
        return state['board_state'] <= 0

    def get_legal_actions_(self, state):
        return tuple(list(range(1, state['board_state']+1))) if self.k > state['board_state'] else tuple(list(range(1, self.k+1)))

    def render_(self, state):
        if self.is_game_over_(state):
            game_over = 'Game is over'
        else:
            game_over = 'Game is not over'
        print(f'Pieces left: {state["board_state"]} \t\t Player to move: {state["pid"]} \t\t {game_over} \t\t '
              f'Legal actions: {self.get_legal_actions_(state)} ')

    def generate_children_(self, state):
        edges = [action for action in self.get_legal_actions_(state)]
        children = [self.step_(state, action) for action in edges]
        return edges, children
