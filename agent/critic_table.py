import copy
import random

import agent.critic


class CriticTable(agent.critic.Critic):
    def __init__(self, learning_rate, discount_factor, trace_decay_fact, seed=None):
        super().__init__()
        self.__learning_rate = learning_rate        # alpha
        self.__discount_factor = discount_factor    # gamma
        self.__trace_decay_fact = trace_decay_fact  # lambda
        self.__td_error = 1                         # delta
        self.__eval = dict()                        # V(s) -> evaluation
        self.__elig = dict()                        # e(s) -> eligibility
        self.__state_current_episode = []
        self.rew = []
        self.disc = []
        self.prev = []
        self.new = []
        if seed:
            random.seed(seed)

    def init_eval(self, state):
        """
        initialize V(s) with small random values.
        :param state:
        :return:
        """
        if state not in self.__eval:
            self.__eval[state] = random.uniform(-0.1, 0.1)

    def update_evals(self):
        for state in self.__state_current_episode:
            self.update_eval(state)

    def update_eval(self, state):
        """
         V(s) ← V(s)+ αδe(s)
        :param state:   s
        :return:
        """
        # self.update_elig(state)
        # self.init_eval(state)
        self.__eval[state] = self.__eval[state] + \
                             self.__learning_rate * \
                             self.__td_error * \
                             self.__elig[state]

    def update_td_error(self, prev_state, new_state, reward, done):
        """
        δ ← r + γV(s') − V(s)
        :param prev_state:  s
        :param new_state:   s'
        :param reward:      rewards received from state transition
        :return:
        """
        # self.update_elig(prev_state)
        # self.update_elig(new_state)
        # self.init_eval(prev_state)
        # self.init_eval(new_state)
        # if done:
        #     self.update_elig(new_state)
        #     print('done')
        self.__td_error = reward + self.__discount_factor * self.__eval[new_state] * (1-int(done)) - self.__eval[prev_state]

    def get_td_error(self):
        return self.__td_error

    def get_delta(self):
        return self.__td_error

    def update_elig(self, state):
        """
         e(s) ← 1 (the critic needs state-based eligibilities)
        :param state:
        :return:
        """
        self.__elig[state] = 1
        if state not in self.__state_current_episode:
            self.__state_current_episode.append(state)

    def decay_eligs(self):
        for state in self.__elig:
            self.decay_elig(state)

    def decay_elig(self, state):
        """
        e(s) ← γλe(s)
        :param state:   s
        :return:
        """
        self.__elig[state] = self.__discount_factor * self.__trace_decay_fact * self.__elig[state]

    def new_episode(self):
        self.__state_current_episode.clear()
        self.__elig = self.__elig.fromkeys(self.__elig, 0)

    def reset_elig(self):
        self.__elig.clear()

    def get_eval(self):
        return self.__eval

    def get_sum_eval(self):
        return sum(self.__eval.values())

    def get_sum_elig(self):
        return sum(self.__elig.values())

    def get_size_current_episode(self):
        return len(self.__state_current_episode)
