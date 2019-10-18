"""Hamiltonian class."""
import numpy as np
import math


class Non_Interaction:
    """Calculate variables regarding the Hamiltonian of given wavefunction."""

    def __init__(self, omega, wavefunction, numerical):
        """Instance of class."""
        self.omega = omega
        self.s = wavefunction
        self.num = numerical

    def laplacian_numerical(self, positions):
        """Numerical differentiation for solving laplacian."""

        step = 0.001
        position_forward = np.array(positions)
        position_backward = np.array(positions)
        psi_current = 0.0
        psi_moved = 0.0

        for i in range(self.s.num_p):
            psi_current += 2*self.s.num_d*self.s.wavefunction(positions)
            for j in range(self.s.num_d):

                position_forward[i, j] = position_forward[i, j] + step
                position_backward[i, j] = position_backward[i, j] - step
                wf_p = self.s.wavefunction(position_forward)
                wf_n = self.s.wavefunction(position_backward)
                psi_moved += wf_p + wf_n
                # Resett positions
                position_forward[i, j] = position_forward[i, j] - step
                position_backward[i, j] = position_backward[i, j] + step

        laplacian = (psi_moved - psi_current)/(step*step)
        return laplacian

    def laplacian_analytical(self, positions):
        """Analytic solution to laplacian for non-interacting case
        and symmetric potential"""
        """Assumes beta = 1.0 and scattering length = a = 0.0"""

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

        laplacian_analytic = -2*d*n*self.s.alpha + 4*(self.s.alpha**2)*c

        return laplacian_analytic

    def trap_potential_energy(self, positions):
        """Return the potential energy of the wavefunction."""
        """omega < 1.0 sylinder"""
        """omega = 1.0 symmetric"""
        """omega > 1.0 elliptic"""
        omega_sq = self.omega*self.omega

        # 0.5*omega_sq*np.sum(np.multiply(positions, positions))
        return 0.5*omega_sq*np.sum(np.multiply(positions, positions))

    def local_energy(self, positions):
        """Return the local energy."""
        if self.num:
            # Run with numerical expression for kinetic energy
            k = (self.laplacian_numerical(positions) /
                 self.s.wavefunction(positions))
        else:
            # Run with analytical expression for kinetic energy
            k = self.laplacian_analytical(positions)

        p = self.trap_potential_energy(positions)
        energy = -0.5*k + p

        return energy

    def quantum_force(self, positions):
        """Return drift force."""
        """This surely is inefficient, rewrite so the quantum force matrix
        gets updated, than calculating it over and over again each time"""
        quantum_force = np.zeros((self.s.num_p, self.s.num_d))
        position_forward = np.array(positions)
        psi_current = self.s.wavefunction(positions)
        psi_moved = 0.0
        step = 0.001

        for i in range(self.s.num_p):
            for j in range(self.s.num_d):
                position_forward[i, j] = position_forward[i, j] + step
                psi_moved = self.s.wavefunction(position_forward)
                # Resett positions
                position_forward[i, j] = position_forward[i, j] - step
                derivative = (psi_moved - psi_current)/step
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
