import copy
import random


class Actor:
    def __init__(self, learning_rate, discount_factor, trace_decay_fact, init_not_greedy_prob, not_greedy_prob_decay_fact):
        self.__learning_rate = learning_rate            # alpha
        self.__discount_factor = discount_factor        # gamma
        self.__trace_decay_fact = trace_decay_fact      # lambda
        self.__not_greedy_prob = init_not_greedy_prob   # epsilon
        self.__not_greedy_prob_decay_fact = not_greedy_prob_decay_fact
        self.__policy = dict()                          # Π(s, a) -> value
        self.__elig = dict()                            # e(s, a) -> eligibility
        self.__chosen_action = None

    def get_chosen_action(self):
        return copy.copy(self.__chosen_action)

    def get_optimal_action(self, state, actions):
        """ Picks action greedily """
        optimal_action = actions[0]
        for action in actions:
            if self.__policy[(state, action)] > self.__policy[(state, optimal_action)]:
                optimal_action = action
        return optimal_action

    def update_chosen_action(self, state, actions):
        """
        a' ← Π(s') the action dictated by the current policy for state s’
        :param state:   s
        :param actions: a
        :return:
        """
        self.init_policy(state, actions)
        if random.uniform(0, 1) < 1 - self.__not_greedy_prob:
            self.__chosen_action = self.get_optimal_action(state, actions)
        else:
            self.__chosen_action = random.choice(actions)

    def init_policy(self, state, actions):
        """
        Initialize  Π(s,a) ← 0 ∀s,a.
        :return:    None
        """
        for action in actions:
            if (state, action) not in self.__policy:
                self.__policy[(state, action)] = 0

    # def init_eligs(self, state, actions):
    #     """
    #     Reset eligibilities in actor: e(s,a) ← 0 ∀s,a
    #     :param state:   state
    #     :param actions: legal actions
    #     :return:
    #     """
    #     for action in actions:
    #         self.init_elig(state, action)

    # def init_elig(self, state, action):
    #     """
    #     Reset eligibilities in actor: e(s,a) ← 0 ∀s,a
    #     :param action:  a
    #     :param state:   state
    #     :return:
    #     """
    #     if (state, action) not in self.__elig:
    #         self.__elig[(state, action)] = 1

    def update_policies(self, delta):
        """
         ∀(s,a) ∈ current episode:
        :param delta: change
        :return:
        """
        for sap in self.__policy:
            self.update_policy(sap[0], sap[1], delta)

    def update_policy(self, state, action, delta):
        """
        Π(s,a) ← Π(s,a) +αδe(s,a)
        :param state:   state
        :param action:  action
        :param delta:   change
        :return:
        """
        if (state, action) in self.__policy:
            self.update_elig(state, action)
            self.__policy[(state, action)] = self.__policy[(state, action)] + \
                                             self.__learning_rate * delta * \
                                             self.__elig[(state, action)]

    def update_elig(self, state, action):
        """
        e(s,a) ← 1 (the actor keeps SAP-based eligibilities)
        :param state:
        :param action:
        :return:
        """
        self.__elig[(state, action)] = 1

    def decay_eligs(self):
        """
         ∀(s,a) ∈ current episode:
        :return:
        """
        for sap in self.__elig:
            self.decay_elig(sap[0], sap[1])

    def decay_elig(self, state, action):
        """
        e(s,a) ← γλe(s,a)
        :param state:   state
        :param action:  action
        :return:
        """
        self.__elig[(state, action)] = self.__discount_factor * \
                                       self.__trace_decay_fact * \
                                       self.__elig[(state, action)]

    def reset_elig(self):
        self.__elig.clear()

    def decay_not_greedy_prob(self):
        self.__not_greedy_prob *= self.__not_greedy_prob_decay_fact

    def get_not_greedy_prob(self):
        return self.__not_greedy_prob

    def get_policy(self):
        return self.__policy

    def get_sum_policy(self):
        return sum(self.__policy.values())

    def get_sum_elig(self):
        return sum(self.__elig.values())
