import copy
import random

import agent.critic


class CriticTable(agent.critic.Critic):
    def __init__(self, learning_rate, discount_factor, trace_decay_fact):
        super().__init__()
        self.__learning_rate = learning_rate        # alpha
        self.__discount_factor = discount_factor    # gamma
        self.__trace_decay_fact = trace_decay_fact  # lambda
        self.__td_error = 1                         # delta
        self.__eval = dict()                        # V(s) -> evaluation
        self.__elig = dict()                        # e(s) -> eligibility

    def init_eval(self, state):
        """
        initialize V(s) with small random values.
        :param state:
        :return:
        """
        if state not in self.__eval:
            self.__eval[state] = random.uniform(0, 3)

    def update_evals(self):
        for state in self.__eval:
            self.update_eval(state)

    def update_eval(self, state):
        """
         V(s) ← V(s)+ αδe(s)
        :param state:   s
        :return:
        """
        self.update_elig(state)
        self.init_eval(state)
        self.__eval[state] = self.__eval[state] + \
                             self.__learning_rate * \
                             self.__td_error * \
                             self.__elig[state]

    def update_td_error(self, prev_state, new_state, reward):
        """
        δ ← r + γV(s') − V(s)
        :param prev_state:  s'
        :param new_state:   s
        :param reward:      rewards received from state transition
        :return:
        """
        self.init_eval(prev_state)
        self.init_eval(new_state)
        self.__td_error = reward + self.__discount_factor * \
                          self.__eval[prev_state] - \
                          self.__eval[new_state]

    def get_td_error(self):
        return copy.copy(self.__td_error)

    def update_elig(self, state):
        """
         e(s) ← 1 (the critic needs state-based eligibilities)
        :param state:
        :return:
        """
        if state not in self.__elig:
            self.__elig[state] = 1

    def decay_eligs(self):
        for state in self.__elig:
            self.decay_elig(state)

    def decay_elig(self, state):
        """
        e(s) ← γλe(s)
        :param state:   s
        :return:
        """
        self.__elig[state] = self.__discount_factor * \
                             self.__trace_decay_fact * \
                             self.__elig[state]

    def reset_elig(self):
        self.__elig.clear()

    def get_eval(self):
        return self.__eval
