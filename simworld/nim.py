class Nim:
    def __init__(self, n, k, init_player=1):
        if not 0 < init_player <= 2:
            raise Exception(f'Init player must be either 1 or 2, not {init_player}')
        self.state = {
            'board_state': n,
            'pid': init_player
                      }
        self.k = k  # max num pieces taken

    def step(self, action):
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
        children = []
        actions = self.get_legal_actions()
        for action in actions:
            dummy_game = self.create_copy()
            dummy_game.step(action)
            children.append(dummy_game)
        return children
