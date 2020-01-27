"""Hamiltonian class."""
import numpy as np
import math


class Lennard_Jones:
    """Calculate variables regarding the Hamiltonian of given wavefunction."""

    def __init__(self, epsilon, sigma, wavefunction):
        """Instance of class."""
        self.epsilon = epsilon
        self.sigma = sigma
        self.s = wavefunction

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
        """Analytical solution to the laplacian"""

        sum = 0.0
        for i in range(self.s.num_p):
            xi = positions[i, 0]
            yi = positions[i, 1]
            zi = positions[i, 2]
            ri = np.array((xi, yi, zi))
            r_i = xi + yi + zi
            for j in range(self.num_p):
                xj = positions[j, 0]
                yj = positions[j, 1]
                zj = positions[j, 2]
                rj = np.array((xj, yj, zj))
                if(j != i):
                    for d in range(self.num_d):
                        r_ij = ri - rj
                        rij = math.sqrt(np.sum((ri - rj)*(ri - rj)))

            term1 = 10*(self.s.alpha*r_i - 2/(rij**2))
            term2 = (25/4)*((self.s.alpha*r_ij)/rij)**8
            sum += term1 + term2
        laplacian = sum

        return laplacian

    def lennard_jones_potential(self, positions):
        """Regular Lennard-Jones potential"""

        V = 0.0
        for i in range(self.num_p):
            for j in range(i, self.num_p-1):
                # ri_minus_rj = np.subtract(positions[i, :], positions[j+1, :])
                r = 0.0
                for k in range(self.num_d):
                    ri_minus_rj = positions[i, k] - positions[j+1, k]
                    r += ri_minus_rj**2
                distance = math.sqrt(r)
                # distance = math.sqrt(np.sum(np.square(ri_minus_rj)))
            C6 = (self.sigma/distance)**6
            C12 = (self.sigma/distance)**12
            V += 4*self.epsilon*(C12 - C6)

        return V

    def local_energy(self, positions):
        """Return the local energy."""
        if self.analytical == 'true':
            # Run with analytical expression for kinetic energy
            k = self.laplacian_analytical(positions)

        else:
            # Run with numerical expression for kinetic energy
            k = (self.laplacian_numerical(positions) /
                 self.s.wavefunction(positions))

        p = self.lennard_jones_potential(positions)
        energy = -0.5*k + p

        return energy
