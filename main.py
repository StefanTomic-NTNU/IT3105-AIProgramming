from simworld.nim import Nim

if __name__ == '__main__':
    sim = Nim(20, 5, 1)

    sim.render()
    sim.step(5)
    sim.render()
    sim.step(5)
    sim.render()
    sim.step(5)
    sim.render()
    sim.step(3)
    sim.render()
    sim.step(2)
    sim.render()
    sim.step(1)
    sim.render()

    # sim.step(4)

    # print(sim.render())
