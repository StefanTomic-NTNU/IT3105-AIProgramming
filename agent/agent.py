from agent.actor import Actor
from agent.critic_nn import CriticNN
from agent.critic_table import CriticTable
import json


class Agent:

    def __init__(self,
                 critic_type,
                 actor_learning_rate,
                 critic_learning_rate,
                 actor_elig_decay_rate,
                 critic_elig_decay_rate,
                 actor_discount_fact,
                 critic_discount_fact,
                 init_not_greedy_prob,
                 not_greedy_prob_decay_fact):
        self.actor = Actor(actor_learning_rate,
                           actor_discount_fact,
                           actor_elig_decay_rate,
                           init_not_greedy_prob,
                           not_greedy_prob_decay_fact
                           )

        if critic_type == 'table':
            self.critic = CriticTable(critic_learning_rate,
                                      critic_discount_fact,
                                      critic_elig_decay_rate)
        elif critic_type == 'nn':
            self.critic = CriticNN()

        else:
            raise Exception('Critic must be of type "table" or "nn"')

    def initialize(self, init_state, init_actions):
        self.critic.init_eval(init_state)
        self.actor.init_policy(init_state, init_actions)

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
        self.actor.decay_not_greedy_prob()
        self.actor.reset_elig()
        self.critic.reset_elig()

    def get_action(self):
        return self.actor.get_chosen_action()
