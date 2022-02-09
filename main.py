import copy
import json
import gym

from environment.pole import Pole
from environment.gambler import Gambler
from environment.hanoi import Hanoi

from agent.agent import Agent


def read_config():
    with open('config.json', 'r') as f:
        config_ = json.load(f)
    return config_


config = read_config()
env = Gambler()
agent = Agent(config['critic_type'],
              config['actor_learning_rate'],
              config['critic_learning_rate'],
              config['actor_elig_decay_rate'],
              config['critic_elig_decay_rate'],
              config['actor_discount_fact'],
              config['critic_discount_fact'],
              config['init_not_greedy_prob'],
              config['not_greedy_prob_decay_fact']
)

nr_episodes = config['nr_episodes']
max_steps = config['max_steps']

for i_episode in range(nr_episodes):
    observation = env.reset()
    chosen_action = None
    for t in range(nr_episodes):
        env.render()
        print(observation)
        prev_state = copy.copy(observation)
        prev_actions = copy.copy(env.action_space)
        chosen_action = env.action_space.sample()

        observation, reward, done, info = env.step(chosen_action)
        new_actions = copy.copy(env.action_space)

        agent.learn(prev_state, prev_actions, chosen_action, reward, observation, new_actions)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
print(env.action_space)
print(env.observation_space)
env.close()

