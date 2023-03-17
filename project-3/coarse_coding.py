
import math
import numpy as np


class CoarseCoding:

    def __init__(self, dimentions, limits, number_tiles):
        self.dimentions = dimentions # How the grid for each tile should look
        self.tiles = number_tiles # Think four since each state is represented by four values
        self.limits = limits # Decide different limits for each tile. The two concerned with angles should be in radians, while the position should be inside a grid (Need two different, not four)


def main():
    dimentions = [5, 5]
    tilings = 5
    limits = [(2*math.pi, 0), (0, 7)]
    offset=lambda n: 2 * np.arange(n) + 1
    print(offset)
    tiling_dims = np.array(np.ceil(len(dimentions)), dtype=np.int) + 1 # Just a number
    offsets = offset(len(dimentions)) * \
      np.repeat([np.arange(4)], len(dimentions), 0).T / float(4) % 1
    print(offsets)
    limit = np.array(limits)
    print(limit)
    norm_dims = np.array(len(dimentions)) / (limit[:, 1] - limit[:, 0])
    print(limit[:, 1])
    print(limit[:, 0])
    print(norm_dims)
    tile_base_ind = np.prod(tiling_dims) * np.arange(tilings)
    print(tile_base_ind)
    print(len(dimentions))
    print(tiling_dims)

    hash_vec = np.array([np.prod(tile_base_ind[0:i]) for i in range(len(dimentions))])
    print(hash_vec)
    n_tiles = tilings * np.prod(tiling_dims)
    print(n_tiles)


if __name__ == "__main__":
    main()
