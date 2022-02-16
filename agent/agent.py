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
                 not_greedy_prob_decay_fact,
                 nn_dims,
                 state_shape,
                 seed=None):
        self.actor = Actor(actor_learning_rate,
                           actor_discount_fact,
                           actor_elig_decay_rate,
                           init_not_greedy_prob,
                           not_greedy_prob_decay_fact,
                           seed=seed)

        if critic_type == 'table':
            self.critic = CriticTable(critic_learning_rate,
                                      critic_discount_fact,
                                      critic_elig_decay_rate,
                                      seed=seed)
        elif critic_type == 'nn':
            self.critic = CriticNN(nn_dims,
                                   state_shape,
                                   critic_learning_rate)
        else:
            raise Exception('Critic must be of type "table" or "nn"')

    # def initialize(self, init_state, init_actions):
    #     self.critic.init_eval(init_state)
    #     self.actor.init_policy(init_state, init_actions)

    def learn(self, prev_state, prev_actions, chosen_action, reward, new_state, new_actions, done):
        # print(prev_state)
        # print(new_state)

        self.critic.init_eval(prev_state)
        self.critic.init_eval(new_state)

        self.actor.update_elig(prev_state, chosen_action)

        self.critic.update_elig(prev_state)
        self.critic.update_td_error(prev_state, new_state, reward, done)

        # ∀(s,a) ∈ current episode:
        self.critic.update_evals()
        self.critic.decay_eligs()

        self.actor.update_policies(self.critic.get_td_error())
        self.actor.decay_eligs()

    def new_episode(self):
        self.actor.decay_not_greedy_prob()
        self.actor.new_episode()
        self.critic.new_episode()

    def get_action(self):
        return self.actor.get_chosen_action()

    def update_chosen_action(self, new_state, new_actions):
        self.actor.update_chosen_action(new_state, new_actions)
