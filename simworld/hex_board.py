import numpy as np


class Cell:
    def __init__(self):
        self.neighbors = []
        self.piece: tuple = (0, 0)


class Hex:
    def __init__(self, size):
        self.pid = 1
        self.size = size
        self.cells = np.empty([size, size], dtype=object)
        for r in range(size):
            for c in range(size):
                new_cell = Cell()
                self.cells[r, c] = new_cell

                self.add_neighbor(new_cell, r-1, c)
                self.add_neighbor(new_cell, r-1, c+1)
                self.add_neighbor(new_cell, r,   c+1)
                self.add_neighbor(new_cell, r+1, c)
                self.add_neighbor(new_cell, r+1, c-1)
                self.add_neighbor(new_cell, r,   c-1)

    def add_neighbor(self, cell, neighbor_r, neighbor_c):
        try:
            cell.neighbors.append(self.cells[neighbor_r, neighbor_c])
        except IndexError:
            pass

    def render(self):

        print_list = []

        for row in range(self.size * 4 - 3):
            print_list.append('')

        for i in range(len(print_list)):
            print_list[i] += '  ' * (abs(round(len(print_list)/2 - i)))

        # print(f'Print List length: {len(print_list)}')
        for r in range(self.size):
            for c in range(self.size):
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
                    print_list[r] += '   \\   /' * (round((len(print_list) - r)/2))
                else:
                    print_list[r] = print_list[r][:-1]
                    print_list[r] += '/   \\   ' * round((r+1) / 2)
        for row in print_list:
            print(row)

    def get_state(self):
        pass


if __name__ == '__main__':
    hex_board = Hex(5)
    hex_board.render()
