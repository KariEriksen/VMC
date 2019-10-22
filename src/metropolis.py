"""Metropolis class."""
import numpy as np
import random
from sampler import Sampler # noqa: 401


class Metropolis:
    """Metropolis methods."""

    # Hamiltonian(omega, step)

    def __init__(self, monte_carlo_steps, delta_R, delta_t, num_particles,
                 num_dimensions, wavefunction, hamiltonian):
        """Instance of class."""
        self.mc_cycles = monte_carlo_steps
        self.delta_R = delta_R
        self.delta_t = delta_t
        self.num_p = num_particles
        self.num_d = num_dimensions
        # self.positions = positions
        self.w = wavefunction
        self.h = hamiltonian
        self.sampler = Sampler(wavefunction, hamiltonian)
        self.c = 0.0

    def metropolis_step(self, positions):
        """Calculate new metropolis step."""
        """with brute-force sampling of new positions."""

        r = random.random()*random.choice((-1, 1))
        # Pick a random particle
        random_index = random.randrange(len(positions))
        new_positions = np.array(positions)
        new_random_position = new_positions[random_index, :]
        # Suggest a new move
        new_positions[random_index, :] = new_random_position + r*self.delta_R
        acceptance_ratio = self.w.wavefunction_ratio(positions, new_positions)
        epsilon = np.random.sample()

        if acceptance_ratio > epsilon:
            positions = new_positions
            self.c += 1.0

        else:
            pass

        return positions, self.c

    def importance_sampling_step(self, positions):
        """Calculate new step with Importance sampling."""
        """With upgrad method for suggetion of new positions."""
        """Given through the Langevin equation.
        D is the diffusion coefficient equal 0.5, xi is a gaussion random
        variable and delta_t is the time step between 0.001 and 0.01"""

        D = 0.5
        F = self.w.quantum_force(positions)
        r = random.random()*random.choice((-1, 1))
        # Pick a random particle and calculate new position
        random_index = random.randrange(len(positions))
        new_positions = np.array(positions)

        term1 = D*F[random_index, :]*self.delta_t
        term2 = r*np.sqrt(self.delta_t)
        new_random_position = new_positions[random_index, :] + term1 + term2
        new_positions[random_index, :] = new_random_position
        prob_ratio = self.w.wavefunction_ratio(positions, new_positions)
        greens_function = self.greens_function(positions, new_positions,
                                               self.delta_t)

        epsilon = np.random.sample()
        acceptance_ratio = prob_ratio*greens_function

        if acceptance_ratio > epsilon:
            positions = new_positions
            self.c += 1.0

        else:
            pass

        return positions, self.c

    def greens_function(self, positions, new_positions, delta_t):
        """Calculate Greens function."""
        greens_function = 0.0

        D = 0.5
        F_old = self.w.quantum_force(positions)
        F_new = self.w.quantum_force(new_positions)
        for i in range(self.s.num_p):
            for j in range(self.s.num_d):
                term1 = 0.5*((F_old[i, j] + F_new[i, j]) *
                             (positions[i, j] - new_positions[i, j]))
                term2 = D*delta_t*(F_old[i, j] - F_new[i, j])
                greens_function += term1 + term2

        greens_function = np.exp(greens_function)

        return greens_function

    def run_metropolis(self, positions):
        """Run the naive metropolis algorithm."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p, self.num_d)
        for i in range(self.mc_cycles):

            new_positions, count = self.metropolis_step(positions)
            positions = new_positions
            self.sampler.sample_values(positions)
        self.sampler.average_values(self.mc_cycles)
        d_El = self.sampler.derivative_energy
        print (self.c)
        return d_El

    def run_importance_sampling(self, positions):
        """Run importance algorithm."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p, self.num_d)
        for i in range(self.mc_cycles):

            new_positions, count = self.importance_sampling_step(positions)
            positions = new_positions
            self.sampler.sample_values(positions)
        self.sampler.average_values(self.mc_cycles)
        d_El = self.sampler.derivative_energy
        return d_El
