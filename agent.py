from actor import Actor
from critic_table import CriticTable
import json


class Agent:

    def __init__(self):
        self.__actor_learning_rate = 0.7    # alpha_a
        self.__critic_learning_rate = 0.7   # alpha_c
        self.__discount_factor = 0.90       # gamma
        self.__trace_decay_fact = 0.90      # lambda
        self.__not_greedy_prob = 0          # epsilon
        self.read_from_config()
        self.actor = Actor(self.__actor_learning_rate,
                           self.__discount_factor,
                           self.__trace_decay_fact,
                           self.__not_greedy_prob)
        self.critic = CriticTable(self.__critic_learning_rate,
                                  self.__discount_factor,
                                  self.__trace_decay_fact)

    def initialize(self, init_state, init_actions):
        self.critic.init_eval(init_state)
        self.actor.init_policy(init_state, init_actions)

    def read_from_config(self):
        pass

    def process_state(self, state):
        return state

    def learn(self, prev_state, prev_actions, chosen_action, reward, new_state, new_actions, done):
        if done:
            self.new_episode()

        self.actor.update_chosen_action(new_state, new_actions)
        self.actor.update_elig(prev_state, prev_actions)

        self.critic.update_td_error(prev_state, new_state, reward)
        self.critic.update_elig(prev_state)

        self.critic.update_evals()
        self.critic.decay_eligs()

        self.actor.update_policies(self.critic.get_td_error())
        self.actor.decay_eligs()

    def new_episode(self):
        self.actor.reset_elig()
        self.critic.reset_elig()

    def get_action(self):
        return self.actor.get_chosen_action()
