import copy
import json
import gym

import matplotlib.pyplot as plt
from gym.spaces import Discrete

from environment.pole import Pole
from environment.gambler import Gambler
from environment.hanoi import Hanoi

from agent.agent import Agent


def read_config():
    with open('config.json', 'r') as f:
        config_ = json.load(f)
    return config_


def process_state(state):
    return state


def process_actions(env_, actions):
    if isinstance(actions, Discrete):
        actions_ = [*range(actions.n)]
        legal_actions = list(map(env_.is_legal_action, actions_))
        return tuple([d for (d, legal_action) in zip(actions_, legal_actions) if legal_action])
    return actions


if __name__ == '__main__':
    scores = []

    config = read_config()

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

    # env = Gambler(0.5)
    env = Hanoi()
    # env = Pole()

    nr_episodes = config['nr_episodes']
    max_steps = config['max_steps']

    for i_episode in range(nr_episodes):
        sum_reward = 0
        observation = env.reset()
        new_state = None
        new_actions = None
        for t in range(max_steps):
            env.render()

            prev_state = copy.copy(observation)
            prev_actions = copy.copy(env.action_space)

            chosen_action = agent.get_action()
            if chosen_action is None:
                chosen_action = env.action_space.sample()
            # else:
            #     chosen_action = Discrete(agent.get_action())

            observation, reward, done, info = env.step(chosen_action)

            sum_reward += reward

            prev_state = process_state(prev_state)
            prev_actions = process_actions(env, prev_actions)
            new_state = process_state(observation)
            new_actions = process_actions(env, env.action_space)

            # print(f'Prev state: {prev_state}')
            # print(f'Prev actions: {prev_actions}')
            # print(f'Chosen action: {chosen_action}')
            # print(f'New state: {new_state}')
            # print(f'New actions: {new_actions}')

            agent.learn(prev_state, prev_actions, reward, new_state, new_actions, done)

            if done:
                print("Episode finished after {} timesteps, with reward {}".format(t + 1, sum_reward))
                break

        scores.append(sum_reward)
        print(f'Episode: {i_episode}')
        print(f'Final state: {new_state}')
        print(f'Final actions: {new_actions}')
        print(f'Final score: {sum_reward}')
        print(f'Epsilon: {agent.actor.get_not_greedy_prob()}')
        # print(f'Policy size: {len(agent.actor.get_policy())}')
        # print(f'Eval size: {len(agent.critic.get_eval())}')
        print('\n\n')
        sum_reward = 0
        agent.new_episode()

    print('\n\n -- ALL EPISODES FINISHED --')
    print(f'Epsilon: {agent.actor.get_not_greedy_prob()}')

    episodes = [*range(nr_episodes)]

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(episodes, scores)  # Plot some data on the axes.
    plt.show()

    env.close()
