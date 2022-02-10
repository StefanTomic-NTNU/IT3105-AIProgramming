import random

import environment.environment


class Gambler(environment.environment.Environment):
    def __init__(self, win_prob):
        super().__init__()
        self.__win_prob = win_prob  # Chance of winning a coin toss

        self.__state = random.randint(0, 99)
        self.__reward = 0
        self.__done = False
        self.__info = None

    def step(self, action):
        if self.__done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        if action > self.__state or action > 100 - self.__state or action <= 0:
            raise Exception(f'Wager {action} was attempted. This is illegal when {self.__state} currency is at hand.')
        if random.uniform(0, 1) < self.__win_prob:
            # Wager won
            self.__state += action
        else:
            self.__state -= action

        self.__reward = action

        if self.__state <= 0:
            self.__done = True
        if self.__state >= 100:
            self.__done = True

        return self.__state, self.__reward, self.__done, self.__info

    def reset(self):
        self.__state = random.randint(0, 99)
        self.__reward = 0
        self.__done = False
        self.__info = None
