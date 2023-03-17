import random
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'Project-3/SimWorld/')
from acrobat import Acrobat

class Agent:

    def __init__(self, episodes):
        self.episodes = episodes


    def visualize_episodes(self, x, y):
        x = np.linspace(0, x, len(y))
        print(x)
        plt.title("Progress of training")
        plt.xlabel("Episode")
        plt.ylabel("Timesteps")
        plt.plot(x, y, marker='o', markersize=5, color="blue")
        plt.show()

def main():
    agent = Agent(5)
    steps = [0]* agent.episodes
    for i in range(agent.episodes):
        print("EPISODE NR ", i)
        print()
        acrobat = Acrobat(6)
        while acrobat.endstate != True: 
            action = random.randint(0, 2)
            action = [-1, 1, 0][action]
            acrobat.move(action)
        steps[i] = acrobat.steps
    agent.visualize_episodes(agent.episodes, steps)

if __name__ == "__main__":
    main()