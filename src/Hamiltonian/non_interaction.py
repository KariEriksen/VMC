"""Hamiltonian class."""
import numpy as np


class Non_Interaction:
    """Calculate variables regarding the Hamiltonian of given wavefunction."""

    def __init__(self, omega, wavefunction, system, analytical):
        """Instance of class."""
        self.omega = omega
        self.w = wavefunction
        self.s = system
        self.analytical = analytical

    def laplacian_numerical(self, positions):
        """Numerical differentiation for solving laplacian."""

        step = 0.001
        position_forward = np.array(positions)
        position_backward = np.array(positions)
        psi_current = 0.0
        psi_moved = 0.0

        for i in range(self.w.num_p):
            psi_current += 2*self.w.num_d*self.w.wavefunction(positions)
            for j in range(self.w.num_d):

                position_forward[i, j] = position_forward[i, j] + step
                position_backward[i, j] = position_backward[i, j] - step
                wf_p = self.w.wavefunction(position_forward)
                wf_n = self.w.wavefunction(position_backward)
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
        d = self.w.num_d
        n = self.w.num_p
        for i in range(self.w.num_p):
            x = positions[i, 0]
            if d == 1:
                c += x**2
            elif d == 2:
                y = positions[i, 1]
                c += x**2 + y**2
            else:
                y = positions[i, 1]
                z = positions[i, 2]
                c += x**2 + y**2 + z**2

        laplacian_analytic = -2*d*n*self.w.alpha + 4*(self.w.alpha**2)*c

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
        if self.analytical == 'true':
            # Run with analyticalnumerical expression for kinetic energy
            k = self.laplacian_analytical(positions)
        else:
            # Run with numerical expression for kinetic energy
            k = (self.laplacian_numerical(positions) /
                 self.w.wavefunction(positions))

        p = self.trap_potential_energy(positions)
        energy = -0.5*k + p

        return energy
