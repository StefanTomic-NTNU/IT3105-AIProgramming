import numpy as np
import math


class TileCoder:
    def __init__(self, tiles_per_dim: list, value_limits: list, tilings: int, offset=lambda n: 4 * np.arange(n) + 1):
        """
        :param tiles_per_dim: Number of tiles for each dimension, as a list
        :param value_limits: Limits of each state variable, as a list of tuples in the form (min, max)
        :param tilings: Total number of tilings
        :param offset: Function for offsetting coordinates
        """
        tiling_dims = np.array(np.ceil(tiles_per_dim), dtype=np.int) + 1

        # Offset of each tile in the form, where each row is a tile, and each column is a dimension
        self.tiling_offsets = offset(len(tiles_per_dim)) * \
                              np.repeat([np.arange(tilings)], len(tiles_per_dim), 0).T / float(tilings) % 1

        # Value limits
        self.value_limits = np.array(value_limits)

        # Tiles / range of values
        self.norm_dims = np.array(tiles_per_dim) / (self.value_limits[:, 1] - self.value_limits[:, 0])

        # Base index in the final string, for each tile
        self.tile_base_index = np.prod(tiling_dims) * np.arange(tilings)

        # [1, 4, 16, 64]
        self.hash_vector = np.array([np.prod(tiling_dims[0:i]) for i in range(len(tiles_per_dim))])

        # Total length of final string
        self.num_tiles = tilings * np.prod(tiling_dims)

    def __getitem__(self, x: np.array):
        y = np.copy(x)
        for i in range(len(x)):
            y[i] = self.value_limits[i, 0] if y[i] < self.value_limits[i, 0] else y[i]
            y[i] = self.value_limits[i, 1] if y[i] > self.value_limits[i, 1] else y[i]
        off_coords = ((y - self.value_limits[:, 0]) * self.norm_dims + self.tiling_offsets).astype(int)
        # binaries = [int(b) for b in bin(off_coords[0])[2:]]
        # print(self._tile_base_ind + np.dot(off_coords, self._hash_vec))
        encoded_state = self.tile_base_index + np.dot(off_coords, self.hash_vector)

        # dtype is unsigned integer of 1 byte. Maybe bool can be used but idk how tf handles that
        final_string = np.zeros(self.num_tiles, dtype='u1')
        for i in encoded_state:
            final_string[i] = 1
        return final_string

    @property
    def n_tiles(self):
        return self.num_tiles


def main():
    t = TileCoder([6, 6, 6, 6], [(0, 2 * math.pi), (4, 6), (0, 2 * math.pi), (3, 7), ], 4)
    print("This ", t[([math.pi], [4.5], [math.pi], [5.2])])


if __name__ == "__main__":
    main()
