import copy
import json
from time import sleep

import numpy as np

import matplotlib.pyplot as plt
from gym.spaces import Discrete

from environment.cartpole import CartPoleEnv
from environment.gambler import Gambler
from environment.hanoi import Hanoi
import tensorflow as tf

from agent.agent import Agent


def read_config():
    with open('config.json', 'r') as f:
        config_ = json.load(f)
    return config_


seed_ = 136
tf.random.set_seed(seed_+6)


pos_bins = 7
cart_vel_bins = 7
angle_bins = 13
pole_vel_bins = 8

def process_state(state, done):
    if isinstance(state, np.ndarray):
        if done:
            return (100, 100, 100, 100)
        # state_ = tuple(map(lambda x: round(x, ndigits=1), state))
        state[0] = bin_state(state[0], -2.4, 2.4, pos_bins)        # Pos
        state[1] = bin_state(state[1], -2, 2, cart_vel_bins)            # Cart velocity
        state[2] = bin_state(state[2], -0.21, 0.21, angle_bins)     # Angle
        state[3] = bin_state(state[3], -2, 2, pole_vel_bins)            # Pole velocity
        state_ = tuple(state)
        return state_
    return state


def bin_state(val, min_, max_, nr_bins):
    bin_ = 0
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


def run(seed):
    step_list = []
    actions = []
    wagers = []
    angles = []
    greedy_steps = []
    for i in range(100):
        wagers.append([])

    config = read_config()
    problem = config['problem']

    env = None
    if problem == 'hanoi':
        env = Hanoi(nr_pegs=config['nr_pegs'], nr_discs=config['nr_discs'])
    if problem == 'cartpole':
        env = CartPoleEnv(pole_length=config['pole_length'],
                          pole_mass=config['pole_mass'],
                          gravity=config['gravity'],
                          timestep=config['timestep']
                          )
        env.seed(seed)
    if problem == 'gambler':
        env = Gambler(win_prob=config['win_prob'])

    state_size = None
    if isinstance(env, Gambler):
        state_size = 98
    if isinstance(env, CartPoleEnv):
        state_size = pos_bins * cart_vel_bins * angle_bins * pole_vel_bins
    if isinstance(env, Hanoi):
        state_size = config['nr_pegs'] * config['nr_discs']

    agent = Agent(config['critic_type'],
                  config['actor_learning_rate'],
                  config['critic_learning_rate'],
                  config['actor_elig_decay_rate'],
                  config['critic_elig_decay_rate'],
                  config['actor_discount_fact'],
                  config['critic_discount_fact'],
                  config['init_not_greedy_prob'],
                  config['not_greedy_prob_decay_fact'],
                  config['nn_dims'],
                  state_size,
                  seed=seed)

    display = config['display']

    nr_episodes = config['nr_episodes'] + 1
    max_steps = config['max_steps']

    steps = 0

    all_state_history = []
    for i_episode in range(nr_episodes):    # Repeat for each episode:
        done = False
        state_history = []

        sum_reward = 0

        agent.new_episode()     # Reset eligibilities in actor and critic: e(s,a) ← 0: e(s) ← 0 ∀s,a

        observation = env.reset()

        prev_state = process_state(observation, done)
        prev_actions = process_actions(env, env.action_space)

        new_state = None

        agent.actor.update_chosen_action(prev_state, prev_actions)
        chosen_action = agent.get_action()  # a_init
        steps_episode = 0

        state_history.append(prev_state)
        for t in range(max_steps):  # Repeat for each step of the episode:

            if display and i_episode == nr_episodes - 1:
                agent.actor.set_not_greedy_prob(0)
                env.render()
                sleep(config['frame_delay'])

            observation, reward, done, info = env.step(chosen_action)

            sum_reward += reward

            new_state = process_state(observation, done)
            state_history.append(new_state)
            new_actions = process_actions(env, env.action_space)

            steps += 1
            steps_episode += 1
            actions.append(chosen_action)
            if isinstance(env, Gambler):
                wagers[prev_state].append(chosen_action)
            if isinstance(env, CartPoleEnv) and i_episode == nr_episodes - 1:
                angles.append(observation[3])
                greedy_steps.append(t)

            if done and not new_actions:
                new_actions = (0,)
            agent.update_chosen_action(new_state, new_actions)
            chosen_action = agent.get_action()

            agent.learn(prev_state, prev_actions, chosen_action, reward, new_state, new_actions, done)

            if done:
                break

            prev_state = copy.copy(new_state)
            prev_actions = copy.copy(new_actions)

        step_list.append(steps_episode)
        print(f'Episode: {i_episode}')
        print(state_history)
        print(f'Final score: {sum_reward}')
        print(f'Epsilon: {agent.actor.get_not_greedy_prob()}')
        state_history.clear()

    episodes = [*range(nr_episodes)]

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(episodes, step_list, label='Steps')  # Plot some data on the axes.
    ax.legend()
    plt.show()

    if isinstance(env, Gambler):
        # Greedy strat:
        agent.actor.set_not_greedy_prob(0)
        greedy_actions = []
        states = range(1, 100)
        for i in range(1, 100):
            env.state = i
            env.update_action_space()
            actions = env.action_space
            greedy_actions.append(agent.actor.get_optimal_action(i, process_actions(env, actions)))

        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(states, greedy_actions, label='Greedy actions')  # Plot some data on the axes.
        ax.legend()
        plt.show()

    env.close()
    return steps


run(seed_)
