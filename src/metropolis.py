"""Metropolis class."""
import numpy as np
import random
import math
from sampler import Sampler # noqa: 401
from data_sampler import *  # noqa: 401


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
        self.c = 0.0

        self.s = Sampler(self.w, self.h)

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

        return positions

    def importance_sampling_step(self, positions, analytic):
        """Calculate new step with Importance sampling."""
        """With upgrad method for suggetion of new positions."""
        """Given through the Langevin equation.
        D is the diffusion coefficient equal 0.5, xi is a gaussion random
        variable and delta_t is the time step between 0.001 and 0.01"""

        D = 0.5
        greens_function = 0.0

        if analytic:
            F_old = self.w.quantum_force(positions)
        else:
            F_old = self.w.quantum_force_numerical(positions)

        r = random.random()*random.choice((-1, 1))
        # Pick a random particle and calculate new position
        random_index = random.randrange(len(positions))
        new_positions = np.array(positions)

        term1 = D*F_old[random_index, :]*self.delta_t
        term2 = r*np.sqrt(self.delta_t)
        new_random_position = new_positions[random_index, :] + term1 + term2
        new_positions[random_index, :] = new_random_position
        prob_ratio = self.w.wavefunction_ratio(positions, new_positions)

        if analytic == 'true':
            F_new = self.w.quantum_force(new_positions)
        else:
            F_new = self.w.quantum_force_numerical(new_positions)

        for i in range(self.num_p):
            for j in range(self.num_d):
                term1 = 0.5*((F_old[i, j] + F_new[i, j]) *
                             (positions[i, j] - new_positions[i, j]))
                term2 = D*self.delta_t*(F_old[i, j] - F_new[i, j])
                greens_function += term1 + term2

        greens_function = np.exp(greens_function)

        epsilon = np.random.sample()
        acceptance_ratio = prob_ratio*greens_function

        if acceptance_ratio > epsilon:
            positions = new_positions
            self.c += 1.0

        else:
            pass

        return positions

    def run_metropolis(self):
        """Run the naive metropolis algorithm."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p, self.num_d)
        # Initialize sampler method for each new Monte Carlo run
        self.s.initialize()

        for i in range(self.mc_cycles):
            new_positions = self.metropolis_step(positions)
            positions = new_positions
            self.s.sample_values(positions)

        self.s.average_values(self.mc_cycles)
        energy = self.s.local_energy
        d_El = self.s.derivative_energy
        self.print_averages()
        return d_El, energy

    def run_importance_sampling(self, analytic):
        """Run importance algorithm."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p, self.num_d)
        # Initialize sampler method for each new Monte Carlo run
        self.s.initialize()

        for i in range(self.mc_cycles):
            new_positions = self.importance_sampling_step(positions, analytic)
            positions = new_positions
            self.s.sample_values(positions)

        self.s.average_values(self.mc_cycles)
        energy = self.s.local_energy
        d_El = self.s.derivative_energy
        self.print_averages()
        return d_El, energy

    def run_one_body_sampling(self):
        """Sample position of particles."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p, self.num_d)
        # Initialize sampler method for each new Monte Carlo run
        self.s.initialize()
        density_adding = np.zeros(41)

        # Run Metropolis while finding one body density
        for i in range(self.mc_cycles):
            new_positions = self.metropolis_step(positions)
            positions = new_positions
            density = self.one_body_density(positions)
            density_adding += density
            # self.s.sample_values(positions)

        # self.s.average_values(self.mc_cycles)

        return density_adding

    def one_body_density(self, positions):
        """Run one-body density count."""

        num_radii = 41
        density = np.zeros(num_radii)
        r_vec = np.linspace(0, 4, num_radii)
        step = r_vec[1] - r_vec[0]

        # Calculate the distance from origo of each particle
        radii = np.zeros(self.num_p)
        for i in range(self.num_p):
            r = 0
            for j in range(self.num_d):
                r += positions[i, j]*positions[i, j]
            radii[i] = math.sqrt(r)

        # Check in which segment each particle is in
        for i in range(self.num_p):
            dr = 0.0
            for j in range(num_radii):
                if(dr <= radii[i] < dr+step):
                    density[j] += 1
                    break
                else:
                    dr += step

        return density

    def print_averages(self):

        print ('acceptence rate = ', self.c/self.mc_cycles)
        print ('new alpha = ', self.w.alpha)
        print ('deri energy = ', self.s.derivative_energy)
        print ('total energy =  ', self.s.local_energy)
        # energy/num_particles
        print ('----------------------------')
