import copy
import random
import time

import numpy as np


class Cell:
    def __init__(self, pos):
        self.neighbors = []
        self.piece: tuple = (0, 0)
        self.pos: tuple = pos


class Hex:
    def __init__(self, size, init_player=None):
        self.SIZE = size
        self.cells = np.empty([size, size], dtype=object)
        for r in range(size):
            for c in range(size):
                new_cell = Cell((r, c))
                self.cells[r, c] = new_cell
        for r in range(size):
            for c in range(size):
                self.add_neighbor(self.cells[r, c], r - 1, c)
                self.add_neighbor(self.cells[r, c], r - 1, c + 1)
                self.add_neighbor(self.cells[r, c], r,     c + 1)
                self.add_neighbor(self.cells[r, c], r + 1, c)
                self.add_neighbor(self.cells[r, c], r + 1, c - 1)
                self.add_neighbor(self.cells[r, c], r,     c - 1)
        if init_player is None:
            init_player = random.randint(1, 2)
        self.state = {'board_state': self.simplify_state(),
                      'pid': init_player}

    def add_neighbor(self, cell, neighbor_r, neighbor_c):
        try:
            if neighbor_r < 0 or neighbor_c < 0:
                return
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
                pieces[r * self.SIZE + c, :] = copy.copy(self.cells[r, c].piece)
        return pieces

    def set_state(self, state):
        self.state = copy.copy(state)   # TODO: Change
        board_state = state['board_state'].copy()
        for i in range(len(board_state)):
            c = i % self.SIZE
            r = np.floor_divide(i, self.SIZE)
            self.cells[r, c].piece = tuple(board_state[i].tolist())

    def get_state(self):
        new_state = {
            'board_state': self.state['board_state'].copy(),
            'pid': copy.copy(self.state['pid'])
        }
        return copy.copy(new_state)

    def get_legal_actions(self):
        return np.argwhere(np.all(self.state['board_state'] == 0, axis=1) == True).ravel()

    def get_legal_actions_(self, state):
        # print(state['board_state'])
        return np.argwhere(np.all(state['board_state'] == 0, axis=1) == True).ravel()

    def get_all_actions(self):
        return np.array(range(self.SIZE ** 2))

    def make_move(self, action):
        if self.is_game_over():
            # print('Game is over')
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
        if self.state is None: return True
        queued_cells = []
        explored_cells = []
        for c in range(self.SIZE):  # Vertical connection for player 1
            cell = self.cells[0, c]
            if cell.piece == (0, 1):
                queued_cells.append(cell)
        while queued_cells:
            cell = queued_cells[0]
            if cell.pos[0] == self.SIZE-1:
                return True
            [queued_cells.append(neighbor) for neighbor in cell.neighbors
             if neighbor and
             neighbor.piece == cell.piece and
             neighbor not in queued_cells and
             neighbor not in explored_cells]
            explored_cells.append(cell)
            queued_cells.pop(0)

        for r in range(self.SIZE):  # Horizontal connection for player 2
            cell = self.cells[r, 0]
            if cell.piece == (1, 0):
                queued_cells.append(cell)
        while queued_cells:
            cell = queued_cells[0]
            if cell.pos[1] == self.SIZE-1:
                return True
            [queued_cells.append(neighbor) for neighbor in cell.neighbors
             if neighbor and
             neighbor.piece == cell.piece and
             neighbor not in queued_cells and
             neighbor not in explored_cells]
            explored_cells.append(cell)
            queued_cells.pop(0)

        return False

    def is_game_over_(self, state):
        board = self.create_copy()
        board.set_state(state)
        return board.is_game_over()

    def create_copy(self):
        game = Hex(self.SIZE, init_player=self.state['pid'])
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                game.cells[r, c].piece = copy.copy(self.cells[r, c].piece)
        game.state['board_state'] = game.simplify_state()
        return game

    def step_(self, state, action):
        new_state = {
            'board_state': state['board_state'].copy(),
            'pid': copy.copy(state['pid'])
        }
        if action not in self.get_legal_actions_(state):
            return
        else:
            piece = (1, 0) if new_state['pid'] == 1 else (0, 1)
            new_state['board_state'][action, :] = np.array(piece)
        new_state['pid'] = 3 - new_state['pid']
        return new_state

    def generate_children_(self, state):
        # edges_s_time = time.time()
        # edges_e_time = time.time()
        # edges_time = edges_e_time - edges_s_time
        #
        # iedges_s_time = time.time()
        # iedges_e_time = time.time()
        # iedges_time = iedges_e_time - iedges_s_time
        #
        # ichild_s_time = time.time()
        # ichild_e_time = time.time()
        # ichild_time = ichild_e_time - ichild_s_time
        #
        # child_s_time = time.time()
        # child_e_time = time.time()
        # child_time = child_e_time - child_s_time

        # print(f'Edges: {edges_time}\t Iedges: {iedges_time}\t Ichild: {ichild_time}\t Child: {child_time}')

        edges = self.get_legal_actions_(state)
        illegal_edges = np.setxor1d(np.array(edges), self.get_all_actions()).tolist()
        illegal_children = [None for _ in illegal_edges]
        children = [self.step_(state, action) for action in edges] if not self.is_game_over() else []
        return edges, children, illegal_edges, illegal_children
