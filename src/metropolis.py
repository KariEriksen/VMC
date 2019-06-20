"""Metropolis class."""
import numpy as np
import random


class Metropolis(self):
    """Docstring."""

    # Sampler(omega, step)

    def __init__(self, delta_R, delta_t, num_particles, num_dimensions,
                 sampler, c):
        """Docstring."""
        self.delta_R = delta_R
        self.delta_t = delta_t
        self.num_p = num_particles
        self.num_d = num_dimensions
        # self.positions = positions
        self.s = sampler
        self.c = c

    def metropolis(self, positions):
        """Run the naive metropolis algorithm."""
        """with brute-force sampling of new positions."""

        # new_positions = new_positions()
        # r = np.random.rand(self.num_p, self.num_d)
        r = random.random()
        # Pick a random particle and suggest a new move
        random_index = random.randrange(len(positions))
        new_positions = positions
        new_random_position = new_positions[random_index, :]
        new_positions[random_index, :] = new_random_position + r*self.delta_R
        acceptance_ratio = self.s.probability(positions, new_positions)
        epsilon = np.random.sample()

        if acceptance_ratio > epsilon:
            positions = new_positions
            self.c += 1.0

        else:
            pass

        energy = self.s.local_energy(positions)

        return energy, positions, self.c

    def importance_sampling(self, positions):
        """Run Importance sampling with upgrad method for suggetion of new
        positions. Given through the Langevin equation.
        D is the diffusion coefficient equal 0.5, xi is a gaussion random
        variable and delta_t is the time step between 0.001 and 0.01"""

        D = 0.5
        F = self.s.drift_force(positions)
        xi = np.random.sampler()
        new_positions = (positions + D*F*self.delta_t +
                         xi*np.sqrt(self.delta_t))

        acceptance_ratio = self.s.greens_function(positions, new_positions)
        epsilon = np.random.sample()

        if acceptance_ratio <= epsilon:
            positions = new_positions

        else:
            pass

        self.s.local_energy(positions)

    def gibbs_sampling(self):
        """Run Gibbs sampling."""
