import numpy as np


class Cell:
    def __init__(self):
        self.neighbors = []


class Hex:
    def __init__(self, size):
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
        for r in range(self.size):
            print_row = ''
            for c in range(self.size):
                print_row += '*\t'

            print_list.append(print_row)

        for row in print_list:
            print(row)


if __name__ == '__main__':
    hex_board = Hex(5)
    hex_board.render()
