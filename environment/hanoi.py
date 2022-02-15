# Code taken from https://github.com/RobertTLange/gym-hanoi
import environment.environment

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import random
import itertools
import numpy as np


class Hanoi(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, nr_pegs=3, nr_discs=2):
        self.num_disks = nr_discs
        self.num_pegs = nr_pegs

        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Tuple(self.num_disks * (spaces.Discrete(3),))

        self.current_state = None
        self.goal_state = self.num_disks * (2,)

        self.done = None
        self.ACTION_LOOKUP = {0: "(0,1) - top disk of pole 0 to top of pole 1 ",
                              1: "(0,2) - top disk of pole 0 to top of pole 2 ",
                              2: "(1,0) - top disk of pole 1 to top of pole 0",
                              3: "(1,2) - top disk of pole 1 to top of pole 2",
                              4: "(2,0) - top disk of pole 2 to top of pole 0",
                              5: "(2,1) - top disk of pole 2 to top of pole 1"}

    def step(self, action):
        """
        * Inputs:
            - action: integer from 0 to 5 (see ACTION_LOOKUP)
        * Outputs:
            - current_state: state after transition
            - reward: reward from transition
            - done: episode state
            - info: dict of booleans (noisy?/invalid action?)
        0. Check if transition is noisy or not
        1. Transform action (0 to 5 integer) to tuple move - see Lookup
        2. Check if move is allowed
        3. If it is change corresponding entry | If not return same state
        4. Check if episode completed and return
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        info = {"transition_failure": False,
                "invalid_action": False}

        move = action_to_move[action]

        if self.move_allowed(move):
            disk_to_move = min(self.disks_on_peg(move[0]))
            moved_state = list(self.current_state)
            moved_state[disk_to_move] = move[1]
            self.current_state = tuple(moved_state)
        else:
            info["invalid_action"] = True

        if self.current_state == self.goal_state:
            reward = 0
            self.done = True
        elif info["invalid_action"] == True:
            reward = -1000
        else:
            reward = -1

        return self.current_state, reward, self.done, info

    def disks_on_peg(self, peg):
        """
        * Inputs:
            - peg: pole to check how many/which disks are in it
        * Outputs:
            - list of disk numbers that are allocated on pole
        """
        return [disk for disk in range(self.num_disks) if self.current_state[disk] == peg]

    def is_legal_action(self, action):
        return self.move_allowed(action_to_move[action])

    def move_allowed(self, move):
        """
        * Inputs:
            - move: tuple of state transition (see ACTION_LOOKUP)
        * Outputs:
            - boolean indicating whether action is allowed from state!
        move[0] - peg from which we want to move disc
        move[1] - peg we want to move disc to
        Allowed if:
            * discs_to is empty (no disc of peg) set to true
            * Smallest disc on target pole larger than smallest on prev
        """
        if move[0] >= self.num_pegs or move[1] >= self.num_pegs:
            return False

        disks_from = self.disks_on_peg(move[0])
        disks_to = self.disks_on_peg(move[1])

        if disks_from:
            return (min(disks_to) > min(disks_from)) if disks_to else True
        else:
            return False

    def reset(self):
        self.current_state = self.num_disks * (0,)
        self.done = False
        return self.current_state

    def render(self, mode='human', close=False):
        rows = []
        # for i in range(self.num_disks):
        #     rows.append([])
        # for j in range(self.num_disks):
        #     rows[j][self.current_state[j]] = '*'
        return

    def set_env_parameters(self, num_disks=4, env_noise=0, verbose=True):
        self.num_disks = num_disks
        self.env_noise = env_noise
        self.observation_space = spaces.Tuple(self.num_disks * (spaces.Discrete(self.num_pegs),))
        self.goal_state = self.num_disks * (self.num_pegs,)

        if verbose:
            print("Hanoi Environment Parameters have been set to:")
            print("\t Number of Disks: {}".format(self.num_disks))
            print("\t Transition Failure Probability: {}".format(self.env_noise))


action_to_move = [(0, 1), (0, 2), (0, 3), (0, 4),
                  (1, 0), (1, 2), (1, 3), (1, 4),
                  (2, 0), (2, 1), (2, 3), (2, 4),
                  (3, 0), (3, 1), (3, 2), (3, 4),
                  (4, 0), (4, 1), (4, 2), (4, 3)]

# action_to_move = {0: (0, 1), 1: (0, 2), 2: (1, 0),
#                   3: (1, 2), 4: (2, 0), 5: (2, 1)}
