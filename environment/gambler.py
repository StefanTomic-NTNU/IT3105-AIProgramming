import random

import environment.environment


class Gambler(environment.environment.Environment):
    def __init__(self, win_prob):
        super().__init__()
        self.__win_prob = win_prob  # Chance of winning a coin toss

        self.__state = random.randint(0, 99)

    def step(self, action):
        pass