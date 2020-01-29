"""System class."""
import math
import numpy as np


class System:
    """Contains distance matrix."""

    # deri_psi = 0.0
    # g        = 0.0
    # f        = 0.0

    def __init__(self, num_particles, num_dimensions):
        """Instance of class."""
        self.num_p = num_particles
        self.num_d = num_dimensions
        self.r_squared = np.zeros(self.num_p)
        self.distances = np.zeros((self.num_p, self.num_p))

    def positions_squared(self, positions):
        """Calculate the distance from origo"""

        for i in range(self.num_p):
            x = positions[i, 0]
            y = positions[i, 1]
            z = positions[i, 2]
            self.r_squared[i] = x*x + y*y + z*z

        return self.r_squared

    def positions_distances(self, positions):
        """Calculate the distances between particles"""

        for i in range(self.num_p):
            for j in range(i, self.num_p-1):
                # ri_minus_rj = np.subtract(positions[i, :], positions[j+1, :])
                r = 0.0
                for k in range(self.num_d):
                    ri_minus_rj = (positions[i, k] -
                                   positions[j+1, k])
                    r += ri_minus_rj**2
                self.distances[i, j+1] = math.sqrt(r)

    def positions_update(self, positions, new_position, i):
        """Update the position matrix for movement of one particle"""
        """particle i = particle number"""

        positions[i, :] = new_position

        return positions

    def distances_update(self, positions, i):
        """Update the distance matrix for movement of one particle"""
        """particle i = particle index"""

        for j in range(self.num_p):
            if j != i:
                r = 0.0
                for k in range(self.num_d):
                    ri_minus_rj = positions[i, k] - positions[j, k]
                    r += ri_minus_rj**2
                self.distances[i, j] = math.sqrt(r)
                self.distances[j, i] = self.distances[i, j]
