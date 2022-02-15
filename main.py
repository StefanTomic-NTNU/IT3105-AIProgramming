import copy
import json
import random
import numpy as np

import gym

import matplotlib.pyplot as plt
from gym.spaces import Discrete

from environment.cartpole import CartPoleEnv
from environment.pole import Pole
from environment.gambler import Gambler
from environment.hanoi import Hanoi

from agent.agent import Agent

import gym


def read_config():
    with open('config.json', 'r') as f:
        config_ = json.load(f)
    return config_


def process_state(state):
    if isinstance(state, np.ndarray):
        # state_ = tuple(map(lambda x: round(x, ndigits=1), state))
        state[0] = bin_state(state[0], -2.4, 2.4, 10)
        state[1] = bin_state(state[1], -3, 3, 8)
        state[2] = bin_state(state[2], -0.418, 0.418, 15)
        state[3] = bin_state(state[3], -3, 3, 8)
        state_ = tuple(state)
        return state_
    return state


def bin_state(val, min_, max_, nr_bins):
    bin_: int = 0
    if val > max_:
        bin_ = nr_bins
    else:
        bin_ = int(round((val - min_) / (max_ - min_) * nr_bins))
    return bin_


def process_actions(env_, actions):
    if isinstance(actions, np.ndarray):
       return tuple(actions.tolist())
    elif isinstance(actions, Discrete):
        actions_ = [*range(actions.n)]
    else:
        actions_ = actions
    legal_actions = list(map(env_.is_legal_action, actions_))
    return tuple([d for (d, legal_action) in zip(actions_, legal_actions) if legal_action])


if __name__ == '__main__':
    scores = []
    deltas = []
    sum_eval = []
    sum_policies = []
    sum_eligs_critic = []
    sum_eligs_actor = []
    len_curr_ep_crit = []
    len_curr_ep_actor = []

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
    # env = Pole()
    env = CartPoleEnv(pole_length=config['pole_length'],
                      pole_mass=config['pole_mass'],
                      gravity=config['gravity'],
                      timestep=config['timestep']
                      )
    # env = Hanoi()

    display = config['display']

    nr_episodes = config['nr_episodes']
    max_steps = config['max_steps']

    steps = 0

    for i_episode in range(nr_episodes):    # Repeat for each episode:
        if i_episode == 998:
            agent.actor.set_not_greedy_prob(0)

        sum_reward = 0

        agent.new_episode()     # Reset eligibilities in actor and critic: e(s,a) ← 0: e(s) ← 0 ∀s,a

        observation = env.reset()

        # Initialize: s ← s_init; a ← Π(s_init)
        # new_state = copy.copy(observation)
        # new_actions = process_actions(env, env.action_space)

        prev_state = process_state(observation)
        prev_actions = process_actions(env, env.action_space)

        new_state = None

        agent.actor.update_chosen_action(prev_state, prev_actions)
        chosen_action = agent.get_action()  # a_init

        for t in range(max_steps):  # Repeat for each step of the episode:
            if display:
                env.render()

            observation, reward, done, info = env.step(chosen_action)

            if done:
                print('done')

            sum_reward += reward

            new_state = process_state(observation)
            # print(new_state)
            new_actions = process_actions(env, env.action_space)

            # print(f'Prev state: {prev_state}')
            # print(f'Prev actions: {prev_actions}')
            # print(f'Chosen action: {chosen_action}')
            # print(f'New state: {new_state}')
            # print(f'New actions: {new_actions}\n')

            agent.update_chosen_action(new_state, new_actions)
            chosen_action = agent.get_action()

            agent.learn(prev_state, prev_actions, chosen_action, reward, new_state, new_actions, done)

            # Plotting shit
            steps += 1
            deltas.append(copy.copy(agent.critic.get_delta()))
            sum_eval.append(copy.copy(agent.critic.get_sum_eval()))
            sum_policies.append(copy.copy(agent.actor.get_sum_policy()))
            sum_eligs_critic.append(copy.copy(agent.critic.get_sum_elig()))
            sum_eligs_actor.append(copy.copy(agent.actor.get_sum_elig()))
            len_curr_ep_crit.append(copy.copy(agent.critic.get_size_current_episode()))
            len_curr_ep_actor.append(copy.copy(agent.actor.get_size_current_episode()))

            prev_state = copy.copy(new_state)
            prev_actions = copy.copy(new_actions)

            if done:
                print("Episode finished after {} timesteps, with reward {}".format(t + 1, sum_reward))
                break

        scores.append(sum_reward)
        print(f'Episode: {i_episode}')
        # print(f'Final state: {new_state}')
        # print(f'Final score: {sum_reward}')
        # print(f'Epsilon: {agent.actor.get_not_greedy_prob()}')
        print(f'Policy size: {len(agent.actor.get_policy())}')
        print(f'Eval size: {len(agent.critic.get_eval())}')
        # print('\n\n')

    print('\n\n -- ALL EPISODES FINISHED --')
    print(f'Epsilon: {agent.actor.get_not_greedy_prob()}')
    # print(agent.critic.get_eval())
    # print(agent.actor.get_policy())

    episodes = [*range(nr_episodes)]
    steps = [*range(steps)]

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(episodes, scores, label='Scores')  # Plot some data on the axes.
    ax.legend()
    plt.show()

    if True:
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(steps, deltas, label='Deltas')  # Plot some data on the axes.
        ax.legend()
        plt.show()

        avg_eval = np.array(sum_eval) / len(agent.critic.get_eval().keys())
        avg_pol = np.array(sum_policies) / len(agent.actor.get_policy().keys())
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(steps, avg_eval, label='Avg eval critic')  # Plot some data on the axes.
        ax.plot(steps, avg_pol, label='Policies actor')  # Plot some data on the axes.
        ax.legend()
        plt.show()

        # fig, ax = plt.subplots()  # Create a figure containing a single axes.
        # ax.plot(steps, sum_eligs_critic, label='Elig critic')  # Plot some data on the axes.
        # ax.plot(steps, sum_eligs_actor, label='Elig actor')  # Plot some data on the axes.
        # ax.legend()
        # plt.show()

        # fig, ax = plt.subplots()  # Create a figure containing a single axes.
        # ax.plot(steps, len_curr_ep_crit, label='Current episode critic')  # Plot some data on the axes.
        # ax.plot(steps, len_curr_ep_actor, label='Current episode actor')  # Plot some data on the axes.
        # ax.legend()
        # plt.show()

    env.close()
