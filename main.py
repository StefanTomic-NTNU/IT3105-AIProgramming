from simworld.nim import Nim

if __name__ == '__main__':
    sim = Nim(20, 5, 1)

    sim.render()
    sim.make_move(5)
    sim.render()
    sim.make_move(5)
    sim.render()
    print(sim.generate_children_(sim.state))
    sim.make_move(5)
    sim.render()
    sim.make_move(3)
    sim.render()
    sim.make_move(2)
    sim.render()
    sim.make_move(1)
    sim.render()
