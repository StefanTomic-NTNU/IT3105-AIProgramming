'''The actor contains the policy'''
import math

from numpy import random


class Actor:

    def __init__(self, config: dict):
        self.learningrate = config['lr']
        self.discountfactor = config['discount_fact']
        self.eligibilitydecayfactor = config['elig_decay_rate']
        self.epsilon = config['init_exploration_rate']
        self.epsilondecayrate = config['exploration_decay_fact']
        self.state_action = {}  # key = (state, action), value = (eligibility, policy)
        random.seed(10)

    def add_sap(self, state, action):
        self.state_action[(tuple(state), action)] = [1, 0]

    def reset_eligibility(self):
        '''Reset the eligibility in the value pair in the state_action dictionary'''
        for state, action in self.state_action:
            self.state_action[(state, action)][0] = 0

    def decay_eligibility(self):
        '''Reset the eligibility in the value pair in the state_action dictionary'''
        for state, action in self.state_action:
            self.state_action[(state, action)][0] *= self.eligibilitydecayfactor

    def decay_exploration_rate(self):
        self.epsilon *= self.epsilondecayrate

    def next_action(self, acro_state, episodeProgress):
        states = []
        possible_actions = [1, 0, -1]
        for key in self.state_action.keys():
            states.append(key[0])
        if acro_state not in states:
            # If the state is not registered before it has to be added to states and made a random choise of action
            for action in possible_actions:
                self.add_sap(acro_state, action)
            choose = random.randint(0, len(possible_actions))
            action = possible_actions[choose]
            return action

        # If the state exist it has to be decided if the actor should exploite vs explore

        # print("THE STATE-ACTION PAIR EXISTED")
        saps = []
        for state in self.state_action.keys():
            if state[0] == acro_state:
                possible_actions.append(state[1])
                saps.append(state)

        # In some cases we want a random choice of action
        exploration = random.uniform(0, 1)
        if exploration < self.epsilon and episodeProgress < 0.9:
            choose = random.randint(0, len(possible_actions))
            return possible_actions[choose]

        # Exploitation
        policy = -math.inf
        action = []
        for sap in saps:
            if self.state_action[sap][1] >= policy:
                policy = self.state_action[sap][1]
                action = sap[1]
        return action

    def update_policy(self, state, action, td_error):
        self.state_action[(state, action)][1] += self.learningrate * td_error * self.state_action[(state, action)][0]

    def update_epsilon(self):
        '''Decrese the epsilon for each episode because we want the agent to perform more exploition'''
        self.epsilon = self.epsilon * self.epsilondecayrate

    # def update_eligibility(self, state, action):
    #     self.state_action[(state, action)][0] = self.discountfactor * self.eligibilitydecayfactor * self.state_action[(state, action)][0]

    # def set_initial_eligibility_value(self):
    #     for sap in self.state_action.keys():
    #         self.state_action[sap][0] = 0
