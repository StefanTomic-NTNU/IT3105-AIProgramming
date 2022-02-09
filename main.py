import copy

import gym as gym

from agent import Agent

env = gym.make('CartPole-v0')
agent = Agent()

for i_episode in range(20):
    observation = env.reset()
    chosen_action = None
    for t in range(100):
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