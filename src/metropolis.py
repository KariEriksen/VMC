"""Metropolis class."""
import numpy as np
import random
import math
from sampler import Sampler # noqa: 401


class Metropolis:
    """Metropolis methods."""

    # Hamiltonian(omega, step)

    def __init__(self, monte_carlo_steps, delta_R, delta_t, num_particles,
                 num_dimensions, wavefunction, hamiltonian, system):
        """Instance of class."""
        self.mc_cycles = monte_carlo_steps
        self.delta_R = delta_R
        self.delta_t = delta_t
        self.num_p = num_particles
        self.num_d = num_dimensions
        # self.positions = positions
        self.w = wavefunction
        self.h = hamiltonian
        self.s = system
        self.c = 0.0

        self.sam = Sampler(self.w, self.h)

    def metropolis_step(self, positions):
        """Calculate new metropolis step."""
        """with brute-force sampling of new positions."""

        # r = random.random()*random.choice((-1, 1))
        # r is a random number drawn from the uniform prob. dist. in [0,1]
        r = random.uniform(-1, 1)
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
            self.s.distances_update(positions, random_index)
            self.c += 1.0

        else:
            pass

        return positions

    def metropolis_step_PBC(self, positions):
        """Calculate new metropolis step."""
        """with brute-force sampling of new positions."""

        # r = random.random()*random.choice((-1, 1))
        # r is a random number drawn from the uniform prob. dist. in [0,1]
        r = random.uniform(-1, 1)
        # Pick a random particle
        random_index = random.randrange(len(positions))
        new_positions = np.array(positions)
        new_random_position = new_positions[random_index, :]
        # Suggest a new move
        new_positions[random_index, :] = new_random_position + r*self.delta_R
        # Check boundarys, apply PBC if necessary
        pbc = self.periodic_boundary_conditions(new_positions, random_index)
        new_positions[random_index, :] = pbc
        acceptance_ratio = self.w.wavefunction_ratio(positions, new_positions)
        epsilon = np.random.sample()

        if acceptance_ratio > epsilon:
            positions = new_positions
            self.s.distances_update_PBC(positions, random_index)
            # print (self.s.distances)
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

        if analytic == 'true':
            F_old = self.w.quantum_force(positions)
        else:
            F_old = self.w.quantum_force_numerical(positions)

        # r = random.random()*random.choice((-1, 1))
        # r = np.random.normal()
        r = random.gauss(0, 1)
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
            self.s.distances_update(positions, random_index)
            self.c += 1.0

        else:
            pass

        return positions

    def run_metropolis(self):
        """Run the naive metropolis algorithm."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p, self.num_d)
        # Initialize the distance matrix
        self.s.positions_distances(positions)
        # Initialize sampler method for each new Monte Carlo run
        self.sam.initialize()

        for i in range(self.mc_cycles):
            new_positions = self.metropolis_step(positions)
            positions = new_positions
            self.sam.sample_values(positions)

        self.sam.average_values(self.mc_cycles)
        energy = self.sam.local_energy
        d_El = self.sam.derivative_energy
        var = self.sam.variance
        self.print_averages()
        return d_El, energy, var

    def run_metropolis_PBC(self):
        """Run the naive metropolis algorithm."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p, self.num_d)
        # Initialize the distance matrix
        self.s.positions_distances_PBC(positions)
        # Initialize sampler method for each new Monte Carlo run
        self.sam.initialize()

        for i in range(self.mc_cycles):
            new_positions = self.metropolis_step_PBC(positions)
            positions = new_positions
            self.sam.sample_values(positions)

        self.sam.average_values(self.mc_cycles)
        energy = self.sam.local_energy
        d_El = self.sam.derivative_energy
        var = self.sam.variance
        # self.print_averages()
        return d_El, energy, var

    def run_importance_sampling(self, analytic):
        """Run importance algorithm."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p, self.num_d)
        # Initialize the distance matrix
        self.s.positions_distances(positions)
        # Initialize sampler method for each new Monte Carlo run
        self.sam.initialize()

        for i in range(self.mc_cycles):
            new_positions = self.importance_sampling_step(positions, analytic)
            positions = new_positions
            self.sam.sample_values(positions)

        self.sam.average_values(self.mc_cycles)
        energy = self.sam.local_energy
        d_El = self.sam.derivative_energy
        var = self.sam.variance
        self.print_averages()
        return d_El, energy, var

    def periodic_boundary_conditions(self, positions, index):
        """Apply periodic boundary conditions"""
        """for the case of strong interaction between particles"""

        moved_particle = positions[index, :]
        x = moved_particle[0]
        y = moved_particle[1]
        z = moved_particle[2]
        # Update L in system
        L = (self.num_p/0.02185)**(1./3.)

        if(x > L):
            moved_particle[0] = x - L
        if(y > L):
            moved_particle[1] = y - L
        if(z > L):
            moved_particle[2] = z - L

        if(x <= 0.0):
            moved_particle[0] = x + L
        if(y <= 0.0):
            moved_particle[1] = y + L
        if(z <= 0.0):
            moved_particle[2] = z + L

        return moved_particle

    def run_one_body_sampling(self):
        """Sample position of particles."""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p, self.num_d)
        # Initialize sampler method for each new Monte Carlo run
        self.sam.initialize()
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

    def blocking(self, analytic):
        """Sample for blocking"""
        """using importance sampling"""

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(self.num_p, self.num_d)
        # Initialize sampler method for each new Monte Carlo run
        self.sam.initialize()
        energy = np.zeros(self.mc_cycles)

        for i in range(self.mc_cycles):
            new_positions = self.importance_sampling_step(positions, analytic)
            positions = new_positions
            self.sam.sample_values(positions)
            energy[i] = self.sam.local_energy

        self.sam.average_values(self.mc_cycles)
        self.print_averages()
        return energy

    def print_averages(self):

        print ('acceptence rate = ', self.c/self.mc_cycles)
        print ('new alpha = ', self.w.alpha)
        print ('deri energy = ', self.sam.derivative_energy)
        print ('total energy =  ', self.sam.local_energy)
        print ('variance energy =  ', self.sam.variance)
        # energy/num_particles
        print ('----------------------------')
