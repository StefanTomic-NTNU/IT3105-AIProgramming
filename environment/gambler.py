import random

import environment.environment


class Gambler(environment.environment.Environment):
    def __init__(self, win_prob):
        super().__init__()
        self.__win_prob = win_prob  # Chance of winning a coin toss

        self.state = random.randint(1, 99)
        self.__reward = 0
        self.__done = False
        self.__info = None
        self.action_space = []
        self.update_action_space()

    def step(self, action):
        if self.__done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        if action > self.state or action > 100 - self.state or action <= 0:
            raise Exception(f'Wager {action} was attempted. This is illegal when {self.state} currency is at hand.')
        if random.uniform(0, 1) < self.__win_prob:
            # Wager won
            self.state += action
            # self.__reward = action
        else:
            self.state -= action
            # self.__reward = -action

        if self.state <= 0:
            self.__done = True
        if self.state >= 100:
            self.__done = True
            self.__reward = 50
        else:
            self.__reward = -1

        self.update_action_space()

        return self.state, self.__reward, self.__done, self.__info

    def update_action_space(self):
        if self.state < 50:
            self.action_space = [*range(1, 1 + self.state)]
        else:
            self.action_space = [*range(1, 101 - self.state)]

    def is_legal_action(self, action):
        return True

    def reset(self):
        self.state = random.randint(1, 99)
        self.update_action_space()
        self.__reward = 0
        self.__done = False
        self.__info = None
        return self.state
