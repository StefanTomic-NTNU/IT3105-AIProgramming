import random

import numpy as np


class Cell:
    def __init__(self):
        self.neighbors = []
        self.piece: tuple = (0, 0)


class Hex:
    def __init__(self, size, init_player=None):
        self.SIZE = size
        self.cells = np.empty([size, size], dtype=object)
        for r in range(size):
            for c in range(size):
                new_cell = Cell()
                self.cells[r, c] = new_cell

                self.add_neighbor(new_cell, r - 1, c)
                self.add_neighbor(new_cell, r - 1, c + 1)
                self.add_neighbor(new_cell, r, c + 1)
                self.add_neighbor(new_cell, r + 1, c)
                self.add_neighbor(new_cell, r + 1, c - 1)
                self.add_neighbor(new_cell, r, c - 1)
        if init_player is None:
            init_player = random.randint(1, 2)
        self.state = {'board_state': self.simplify_state(),
                      'pid': init_player}

    def add_neighbor(self, cell, neighbor_r, neighbor_c):
        try:
            cell.neighbors.append(self.cells[neighbor_r, neighbor_c])
        except IndexError:
            pass

    def render(self):

        print_list = []

        for row in range(self.SIZE * 4 - 3):
            print_list.append('')

        for i in range(len(print_list)):
            print_list[i] += '  ' * (abs(round(len(print_list) / 2 - i)))

        # print(f'Print List length: {len(print_list)}')
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                print_index = (r + c) * 2

                if self.cells[r, c].piece == (0, 0):
                    symbol = '*'
                elif self.cells[r, c].piece == (1, 0):
                    symbol = 'o'
                elif self.cells[r, c].piece == (0, 1):
                    symbol = 'x'
                else:
                    raise Exception

                if '*' in print_list[print_index] \
                        or 'o' in print_list[print_index] \
                        or 'x' in print_list[print_index]:
                    print_list[print_index] += f'----- {symbol} '
                else:
                    print_list[print_index] += f' {symbol} '

        for r in range(len(print_list)):
            if r % 2 == 1:
                if r > len(print_list) / 2:
                    print_list[r] += '   \\   /' * (round((len(print_list) - r) / 2))
                else:
                    print_list[r] = print_list[r][:-1]
                    print_list[r] += '/   \\   ' * round((r + 1) / 2)
        for row in print_list:
            print(row)

    def simplify_state(self):
        pieces = np.empty((self.SIZE ** 2, 2))
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                pieces[r * self.SIZE + c, :] = self.cells[r, c].piece
        return pieces

    def get_legal_actions(self):
        return np.argwhere(np.all(self.state['board_state'] == 0, axis=1) == True).ravel()

    def make_move(self, action):
        if self.is_game_over():
            print('Game is over')
            return
        if action not in self.get_legal_actions():
            return
        else:
            c = action % self.SIZE
            r = np.floor_divide(action, self.SIZE)
            piece = (1, 0) if self.state['pid'] == 1 else (0, 1)
            self.state['board_state'][action, :] = np.array(piece)
            self.cells[r, c].piece = piece
        self.state['pid'] = 3 - self.state['pid']

    def is_game_over(self):
        if self.state is None:
            return True


if __name__ == '__main__':
    hex_board = Hex(5)
    hex_board.render()
