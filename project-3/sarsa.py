import math

import numpy as np
from matplotlib import pyplot as plt

from model import Critic_NN
from actor import Actor
from SimWorld.acrobat import Acrobat, Animate
from timeit import timeit as timer


def run_algo(config: dict):
    debug_info = config['sarsa']['debug_info']
    if debug_info: algo_start_time = timer()
    actor = Actor(config["actor"])
    critic = Critic_NN(config["critic"])
    episodes = config['sarsa']['episodes']
    gen_graphs = config['sarsa']['gen_graphs']  # weather or not to log data and display graphs

    if gen_graphs:
        episodes_x = []
        steps_count = 0

        # Episode basis:
        total_steps_taken = []
        nr_saps_explored = []
        avg_elig = []
        avg_value = []
        td_errors = []

    for episode in range(episodes):
        if episode == episodes - 1: actor.epsilon = 0.00
        print(f'\nEpisode: {episode+1} / {episodes}')
        print(f'Exploration rate: {actor.epsilon:.3f}')
        print(f'Total Number of saps explored: {len(actor.state_action)}')
        acrobat = Acrobat(config["simworld"])
        state = acrobat.get_state()
        action = actor.next_action(state, episode / episodes)

        while acrobat.steps < config['sarsa']['max_steps'] and not acrobat.endstate:
            new_state, reward = acrobat.move(action)
            if acrobat.endstate:
                pass

            new_action = actor.next_action(new_state, episode / episodes)
            actor.add_sap(new_state, new_action)
            # print(new_state)

            td_error = critic.custom_fit_2(state, new_state, reward)
            # td_error = critic.get_TD_error(state, action, new_state, new_action, reward)

            # critic.customFit(state, action, td_error)
            # critic.update_eligibility()

            for state, action in actor.state_action:
                actor.update_policy(state, action, td_error)

            actor.decay_eligibility()

            state = new_state
            action = new_action

            if gen_graphs:
                steps_count += 1
                td_errors.append(td_error)

        actor.decay_exploration_rate()
        actor.reset_eligibility()

        if config['simworld']['animate'] and acrobat.steps < 150:
            print('Animating.. ')
            animat = Animate(acrobat.animate_x, acrobat.animate_y, acrobat.steps)
            animat.animate_acro(use_steps=True)

        print(f'Steps taken: {acrobat.steps}')
        print(f'Min velocity1: {acrobat.min_velocity1}')
        print(f'Max velocity1: {acrobat.max_velocity1}')
        print(f'Min velocity2: {acrobat.min_velocity2}')
        print(f'Max velocity2: {acrobat.max_velocity2}')
        if gen_graphs:
            episodes_x.append(episode)
            total_steps_taken.append(acrobat.steps)
            nr_saps_explored.append(len(actor.state_action))
            eligs = []
            values = []
            for entry in actor.state_action.values():
                eligs.append(entry[0])
                values.append(entry[1])
            avg_elig.append(sum(eligs) / len(eligs))
            avg_value.append(sum(values) / len(values))

    runtime = timer() - algo_start_time
    minutes = math.floor(runtime / 60)
    seconds = runtime % 60
    if debug_info: print(f'\nTotal runtime: {minutes} minutes, {seconds:.1f} seconds')

    if config['simworld']['animate']:
        print('Animating.. ')
        animat = Animate(acrobat.animate_x, acrobat.animate_y, acrobat.steps)
        animat.animate_acro()

    if gen_graphs:
        print('Generating graphs.. ')
        steps_x = [*range(steps_count)]

        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(episodes_x, total_steps_taken, label='Steps in episode')  # Plot some data on the axes.
        ax.legend()
        plt.show()

        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(episodes_x, nr_saps_explored, label='Saps explored')  # Plot some data on the axes.
        ax.legend()
        plt.show()

        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(episodes_x, avg_elig, label='Avg elig')  # Plot some data on the axes.
        ax.plot(episodes_x, avg_value, label='Avg value')  # Plot some data on the axes.
        ax.legend()
        plt.show()

        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(steps_x, td_errors, label='TD error')  # Plot some data on the axes.
        ax.legend()
        plt.show()
