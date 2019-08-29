"""Sampler class."""
import math
import numpy as np


class Sampler:
    """Calculate variables regarding energy of given system."""

    def __init__(self, omega, numerical_step, system):
        """Instance of class."""
        self.omega = omega
        self.step = numerical_step
        self.s = system

    def kinetic_energy(self, positions):
        """Numerical differentiation for solving kinetic energy."""
        """The -0.5 part is added in local energy function"""
        # kine_energy = 0.0

        position_forward = np.array(positions)
        position_backward = np.array(positions)
        psi_current = 0.0
        psi_moved = 0.0

        for i in range(self.s.num_p):
            psi_current += 2*self.s.num_d*self.s.wavefunction(positions)
            for j in range(self.s.num_d):

                position_forward[i, j] = position_forward[i, j] + self.step
                position_backward[i, j] = position_backward[i, j] - self.step
                wf_p = self.s.wavefunction(position_forward)
                wf_n = self.s.wavefunction(position_backward)
                psi_moved += wf_p + wf_n
                # Resett positions
                position_forward[i, j] = position_forward[i, j] - self.step
                position_backward[i, j] = position_backward[i, j] + self.step

        kine_energy = (psi_moved - psi_current)/(self.step*self.step)
        # kine_energy = kine_energy/self.s.wavefunction(positions)

        return kine_energy

    def kinetic_analytic(self, positions):
        """Assumes beta = 1.0"""
        """The -0.5 part is added in local energy function"""

        c = 0.0
        d = self.s.num_d
        n = self.s.num_p
        for i in range(self.s.num_p):
            x = positions[i, 0]
            y = positions[i, 1]
            if d > 2:
                z = positions[i, 2]
                c += x**2 + y**2 + z**2
            else:
                c += x**2 + y**2

        kine_energy_analytic = -2*d*n*self.s.alpha + 4*(self.s.alpha**2)*c

        return kine_energy_analytic

    def potential_energy(self, positions):
        """Return the potential energy of the system."""
        omega_sq = self.omega*self.omega

        # 0.5*omega_sq*np.sum(np.multiply(positions, positions))
        return 0.5*omega_sq*np.sum(np.multiply(positions, positions))

    def interaction(self, positions):
        """Return the interaction potential between particles"""
        internal_potential = 0.0
        for i in range(self.s.num_p):
            for j in range(i, self.s.num_p-1):
                ri_minus_rj = np.subtract(positions[i, :], positions[j+1, :])
                distance = math.sqrt(np.sum(np.square(ri_minus_rj)))

                if distance > self.s.a:
                    internal_potential += 0.0
                else:
                    internal_potential += 1e10

        return internal_potential

    def local_energy(self, positions):
        """Return the local energy."""
        # Run with analytical expression for kinetic energy
        # k = self.kinetic_analytic(positions)
        # Run with numerical expression for kinetic energy
        k = self.kinetic_energy(positions)/self.s.wavefunction(positions)
        p = self.potential_energy(positions)
        # i = self.interaction(positions)
        energy = -0.5*k + p

        return energy

    def local_energy_times_wf(self, positions):
        """Return local energy times the derivative of wave equation."""
        energy = self.local_energy(positions)
        energy_times_wf = self.s.derivative_psi_term(positions)*energy

        return energy_times_wf

    def probability(self, positions, new_positions):
        """Wave function with new positions squared divided by."""
        """wave equation with old positions squared"""
        wf_old = self.s.wavefunction(positions)
        wf_new = self.s.wavefunction(new_positions)
        numerator = wf_new*wf_new
        denominator = wf_old*wf_old
        acceptance_ratio = numerator/denominator

        return acceptance_ratio

    def quantum_force(self, positions):
        """Return drift force."""

        """This surely is inefficient, rewrite so the quantum force matrix
        gets updated, than calculating it over and over again each time"""
        quantum_force = np.zeros((self.s.num_p, self.s.num_d))
        position_forward = np.array(positions)
        psi_current = self.s.wavefunction(positions)
        psi_moved = 0.0

        for i in range(self.s.num_p):
            for j in range(self.s.num_d):
                position_forward[i, j] = position_forward[i, j] + self.step
                psi_moved = self.s.wavefunction(position_forward)
                # Resett positions
                position_forward[i, j] = position_forward[i, j] - self.step
                derivative = (psi_moved - psi_current)/self.step
                quantum_force[i, j] = (2.0/psi_current)*derivative

        return quantum_force

    def greens_function(self, positions, new_positions, delta_t):
        """Calculate Greens function."""
        greens_function = 0.0

        D = 0.5
        F_old = self.quantum_force(positions)
        F_new = self.quantum_force(new_positions)
        for i in range(self.s.num_p):
            for j in range(self.s.num_d):
                term1 = 0.5*((F_old[i, j] + F_new[i, j]) *
                             (positions[i, j] - new_positions[i, j]))
                term2 = D*delta_t*(F_old[i, j] - F_new[i, j])
                greens_function += term1 + term2

        greens_function = np.exp(greens_function)

        return greens_function
