"""System class."""
import math
import numpy as np


class System:
    """Contains position matrix."""

    # deri_psi = 0.0
    # g        = 0.0
    # f        = 0.0

    def __init__(self, num_particles, num_dimensions):
        """Instance of class."""
        self.num_p = num_particles
        self.num_d = num_dimensions
        self.positions = np.random.rand(self.num_p, self.num_d)
        self.r_squared = np.zeros(self.num_p)
        self.distances = np.zeros((self.num_p, self.num_p))

    def positions_squared(self):
        """Calculate the distance from origo"""

        for i in range(self.num_p):
            x = self.positions[i, 0]
            y = self.positions[i, 1]
            z = self.positions[i, 2]
            self.r_squared[i] = x*x + y*y + z*z

        return self.r_squared

    def positions_distances(self):
        """Calculate the distances between particles"""

        for i in range(self.num_p):
            for j in range(i, self.num_p-1):
                # ri_minus_rj = np.subtract(positions[i, :], positions[j+1, :])
                for k in range(self.num_d):
                    ri_minus_rj = (self.positions[i, k] -
                                   self.positions[j+1, k])
                    r = ri_minus_rj**2
                self.distances[i, j+1] = math.sqrt(r)
        print (self.distances)

        return self.distances

    def positions_update(self, i):
        """Update the distance matrix for movement of one particle"""

        for j in range(self.num_p):
            for k in range(self.num_d):
                ri_minus_rj = self.positions[i, k] - self.positions[j, k]
                r = ri_minus_rj**2
            self.distances[i, j] = math.sqrt(r)
            # self.distances[j, i] = self.distances[i, j]

        return 0
