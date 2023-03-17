'''Four possible states and three possible actions (+1, -1, 0).
Pair of segments of length L connected in J2 (Active), while one connected to wall at J1 (Passive).
F is applied.
Each episode starts with both segments vertical and both angular velocities=0. Is finished when lower segment reaches a target height.
'''
import math
import random
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.animation import FuncAnimation
import sys

sys.path.insert(0, 'Project-3/')
from coarse_coder import TileCoder


class Acrobat:
    def __init__(self, config: dict) -> None:
        self.angle_1 = 0  # Angle for J1
        self.angle_2 = 0  # Angle for J2
        self.velocity_1 = 0  # Angular velocity J1
        self.velocity_2 = 0  # Angular velocity J2
        self.goal_height = config['goal_height']
        self.endstate = False
        self.animate_acrobat = config['animate']

        self.m = config['mass']  # Mass of segments
        self.l = config['length']  # Length of segments
        self.l_end_mass = config['length_end_mass']  # Length from endpoint to center-of-mass
        self.timestep = config['timestep']
        self.g = config['gravity']  # Gravity
        self.phi_1 = 0
        self.phi_2 = 0
        self.d_1 = 0
        self.d_2 = 0
        self.acceleration_1 = 0
        self.acceleration_2 = 0
        self.x_1 = 0
        self.y_1 = 5
        self.steps = 0
        # self.Coarse = CoarseCoding(5, 1, 3)
        self.animate_x = []
        self.animate_y = []
        self.reward = 0
        self.max_velocity1 = 0
        self.max_velocity2 = 0
        self.min_velocity1 = 0
        self.min_velocity2 = 0
        self.tiles = TileCoder(config['tiles_per_dim'], [(0, 2 * math.pi), (-7, 7), (0, 2 * math.pi), (-13, 13)],
                               config['tilings'])  # Find out what the limitations should be

    def encode(self):
        angle1 = self.angle_1 if self.angle_1 > 0 else 2*math.pi - self.angle_1
        angle1 = angle1 % (2*math.pi)
        angle2 = self.angle_2 if self.angle_2 > 0 else 2*math.pi - self.angle_2
        angle2 = angle2 % (2*math.pi)
        self.max_velocity1 = self.velocity_1 if self.velocity_1 > self.max_velocity1 else self.max_velocity1
        self.min_velocity1 = self.velocity_1 if self.velocity_1 < self.min_velocity1 else self.min_velocity1
        self.max_velocity2 = self.velocity_2 if self.velocity_2 > self.max_velocity2 else self.max_velocity2
        self.min_velocity2 = self.velocity_2 if self.velocity_2 < self.min_velocity2 else self.min_velocity2
        return self.tiles[np.array([angle1, self.velocity_1, angle2, self.velocity_2])]

    def is_endstate(self, y_tip):
        if y_tip >= self.goal_height:
            return True
        return False

    def get_reward(self):
        return self.reward

    # constitute intermediate terms
    def update_parameters(self, action):
        self.phi_2 = self.m * self.l_end_mass * self.g * math.cos(self.angle_1 + self.angle_2 - math.pi / 2)
        self.phi_1 = -self.m * self.l * self.l_end_mass * self.velocity_2 ** 2 * math.sin(
            self.angle_2) - 2 * self.m * self.l * self.l_end_mass * self.velocity_1 * self.velocity_2 * math.sin(
            self.angle_2) + (self.m * self.l_end_mass + self.m * self.l) * self.g * math.cos(
            self.angle_1 - math.pi / 2) + self.phi_2
        self.d_2 = self.m * (self.l_end_mass ** 2 + self.l * self.l_end_mass * math.cos(self.angle_2)) + 1
        self.d_1 = self.m * self.l_end_mass ** 2 + self.m * (
                self.l ** 2 + self.l_end_mass ** 2 + 2 * self.l * self.l_end_mass * math.cos(self.angle_2)) + 2
        self.acceleration_2 = (self.m * self.l_end_mass ** 2 + 1 - ((self.d_2 ** 2) / self.d_1)) ** (-1) * (action + (
                self.d_2 / self.d_1) * self.phi_1 - self.m * self.l * self.l_end_mass * self.velocity_1 ** 2 * math.sin(
            self.angle_2) - self.phi_2)
        self.acceleration_1 = - (self.d_2 * self.acceleration_2 + self.phi_1) / self.d_1

    def update_state(self):
        self.velocity_2 = self.velocity_2 + self.timestep * self.acceleration_2
        self.velocity_1 = self.velocity_1 + self.timestep * self.acceleration_1
        self.angle_2 = self.angle_2 + self.timestep * self.velocity_2
        self.angle_1 = self.angle_1 + self.timestep * self.velocity_1
        # print("Angle one ", self.angle_1)
        # print("Angle two ", self.angle_2)

    def endpoints(self):
        angle_3 = self.angle_1 + self.angle_2
        x_2 = self.x_1 + self.l * math.sin(self.angle_1)
        y_2 = self.y_1 - self.l * math.cos(self.angle_1)
        x_tip = x_2 + self.l * math.sin(angle_3)
        y_tip = y_2 - self.l * math.cos(angle_3)
        return [x_2, y_2, x_tip, y_tip]

    def move(self, action):
        self.update_parameters(action)
        self.update_state()
        # print(self.encode())
        endpoints = self.endpoints()
        self.steps += 1
        if self.animate_acrobat:
            self.animate_x.append([self.x_1, endpoints[0], endpoints[2]])
            self.animate_y.append([self.y_1, endpoints[1], endpoints[3]])
        # print("Performing action ", action, " the tip of the acrobat has reached ", endpoints[3])
        if self.is_endstate(endpoints[3]):
            for i in range(10):
                self.animate_x.append([self.x_1, endpoints[0], endpoints[2]])
                self.animate_y.append([self.y_1, endpoints[1], endpoints[3]])
            self.endstate = True
            print("The acrobat has reached its goal height")
            reward = self.goal_height * 100
        else:
            reward = - (3 / (abs(endpoints[3]) + 1))
        return self.get_state(), reward

    def get_state(self) -> tuple:
        return tuple(self.encode().tolist())
        # return [self.angle_1, self.velocity_1, self.angle_2, self.velocity_2]


class Animate:
    def __init__(self, x_values, y_values, steps):
        self.fig, ax = plt.subplots(figsize=(8, 6))
        plt.text(-0.38, 6.2, 'Goal Height', color='w', fontdict={'size': 14, 'weight': 'bold'})
        plt.fill_between([-2, 2], 6, 7.1, color='#3CB371')
        plt.fill_between([-2, 2], 2.8, 6, color='#FFFFF0')
        plt.yticks(color='w')
        plt.tick_params(bottom=False, left=False)
        plt.xticks(color='w')
        plt.plot([-2, 2], [6, 6], '#000000')
        plt.plot(0, 5, marker='o', color='#000000', markersize=8)
        ax.set(xlim=(-2, 2), ylim=(2.8, 7.1))
        self.t = np.linspace(0, 100, len(x_values))
        self.y = y_values
        self.x = x_values
        self.line = ax.plot(self.x[0], self.y[0], color='k', lw=2)[0]
        self.steps = steps

    def animate_lines(self, i):
        self.line.set_ydata(self.y[i])
        self.line.set_xdata(self.x[i])
        if i > self.steps:
            plt.title("The acrobat reached its goal", fontdict={'size': 18, 'weight': 'bold'})
        else:
            plt.title("Step " + str(i), fontdict={'size': 18, 'weight': 'bold'})

    def animate_acro(self, use_steps=False):
        anim = FuncAnimation(
            self.fig, self.animate_lines, interval=80, frames=len(self.t) - 1)
        if use_steps:
            anim.save(f'animations/result{str(self.steps)}.gif')
        plt.draw()
        plt.show()


def main():
    with open('../config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    acrobat = Acrobat(config)
    # acrobat.animate_acrobat = True
    while acrobat.endstate != True:
        action = random.randint(0, 2)
        action = [-1, 1, 0][action]
        acrobat.move(action)
    print("The number of moves ", acrobat.steps)
    animat = Animate(acrobat.animate_x, acrobat.animate_y, acrobat.steps)
    animat.animate_acro()


if __name__ == "__main__":
    main()
