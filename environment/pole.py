import gym

import environment.environment


class Pole(environment.environment.Environment):
    def __init__(self):
        super().__init__()
        self.__gym_env = gym.make('CartPole-v0')
        self.observation_space = self.__gym_env.observation_space
        self.action_space = self.__gym_env.action_space

    def step(self, action):
        return self.__gym_env.step(action)

    def is_legal_action(self, action):
        return True

    def render(self):
        self.__gym_env.render()

    def close(self):
        self.__gym_env.close()

    def reset(self):
        self.__gym_env.reset()
