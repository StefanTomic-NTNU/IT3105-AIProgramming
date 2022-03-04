class Nim:
    def __init__(self, n, k, init_player=1):
        # self.state = [n, init_player]   # [board_state, pid]
        self.n = n  # num pieces on the board
        self.k = k  # max num pieces taken
        if 0 < init_player <= 2:
            self.player = init_player
        else:
            raise Exception(f'Init player must be either 1 or 2, not {init_player}')

    def step(self, action):
        if self.is_game_over():
            print('Game is over')
            return
        if action > self.k:
            raise Exception(f'Cannot take {action} pieces, when {self.k} is the max')
        if action > self.n:
            self.n = 0
        else:
            self.n -= action
        self.player = 3 - self.player

    def is_game_over(self):
        return self.n <= 0

    def get_legal_actions(self):
        return tuple(list(range(1, self.n+1))) if self.k > self.n else tuple(list(range(1, self.k+1)))

    def render(self):
        if self.is_game_over():
            game_over = 'Game is over'
        else:
            game_over = 'Game is not over'
        print(f'Pieces left: {self.n} \t\t Player to move: {self.player} \t\t {game_over} \t\t '
              f'Legal actions: {self.get_legal_actions()} ')
