import copy
import random


class Nim:
    def __init__(self, n, k, init_player=None):
        if init_player is None:
            init_player = random.randint(1, 2)

        if not 0 < init_player <= 2:
            raise Exception(f'Init player must be either 1 or 2, not {init_player}')

        self.state = {
            'board_state': n,
            'pid': init_player
                      }
        self.K = k  # max num pieces taken

    def make_move(self, action):
        if self.is_game_over():
            print('Game is over')
            return
        if action > self.K:
            raise Exception(f'Cannot take {action} pieces, when {self.K} is the max')
        if action > self.state['board_state']:
            self.state['board_state'] = 0
        else:
            self.state['board_state'] -= action
        self.state['pid'] = 3 - self.state['pid']

    def is_game_over(self):
        if self.state is None:
            return True
        return self.state['board_state'] <= 0

    def set_state(self, state):
        self.state = copy.copy(state)

    def get_legal_actions(self):
        return tuple(list(range(1, self.state['board_state']+1))) if self.K > self.state['board_state'] else tuple(list(range(1, self.K + 1)))

    def get_all_actions(self):
        return tuple(list(range(1, self.K + 1)))

    def render(self):
        if self.is_game_over():
            game_over = 'Game is over'
        else:
            game_over = 'Game is not over'
        print(f'Pieces left: {self.state["board_state"]} \t\t Player to move: {self.state["pid"]} \t\t {game_over} \t\t '
              f'Legal actions: {self.get_legal_actions()} ')

    def create_copy(self):
        return Nim(self.state['board_state'], self.K, init_player=self.state['pid'])

    def generate_children(self):
        edges = []
        illegal_edges = []
        children = []
        illegal_children = []   # O_o
        actions = self.get_legal_actions()
        all_actions = self.get_all_actions()
        illegal_actions = set(actions) ^ set(all_actions)
        for action in actions:
            dummy_game = self.create_copy()
            dummy_game.make_move(action)
            edges.append(action)
            children.append(dummy_game.state)
        for illegal_action in illegal_actions:
            illegal_edges.append(illegal_action)
            illegal_children.append(None)
        return edges, children, illegal_edges, illegal_children

    def step_(self, state, action):
        if self.is_game_over_(state):
            print('Game is over')
            return
        new_state = {
            'board_state': state['board_state'],
            'pid': state['pid']
                      }
        if action > self.K:
            raise Exception(f'Cannot take {action} pieces, when {self.K} is the max')
        if action > state['board_state']:
            new_state['board_state'] = 0
        else:
            new_state['board_state'] -= action
        new_state['pid'] = 3 - state['pid']
        return new_state

    def is_game_over_(self, state):
        return state['board_state'] <= 0

    def get_legal_actions_(self, state):
        return tuple(list(range(1, state['board_state']+1))) if self.K > state['board_state'] else tuple(list(range(1, self.K + 1)))

    def is_action_legal_(self, state, action):
        if state is None: return False
        legal_actions = self.get_legal_actions_(state)
        print(legal_actions)
        return action in legal_actions

    def render_(self, state):
        if self.is_game_over_(state):
            game_over = 'Game is over'
        else:
            game_over = 'Game is not over'
        print(f'Pieces left: {state["board_state"]} \t\t Player to move: {state["pid"]} \t\t {game_over} \t\t '
              f'Legal actions: {self.get_legal_actions_(state)} ')

    def generate_children_(self, state):
        edges = [action for action in self.get_legal_actions_(state)]
        illegal_edges = list(set(edges) ^ set(self.get_all_actions()))
        illegal_children = [None for _ in illegal_edges]
        children = [self.step_(state, action) for action in edges]
        return edges, children, illegal_edges, illegal_children
