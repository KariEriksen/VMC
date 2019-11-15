"""Wavefunction class."""
import math
import numpy as np


class McMillian_Wavefunction:
    """Contains parameters of wavefunction and wave equation."""

    # deri_psi = 0.0
    # g        = 0.0
    # f        = 0.0

    def __init__(self, num_particles, num_dimensions, alpha, beta, a):
        """Instance of class."""
        self.num_p = num_particles
        self.num_d = num_dimensions
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.k = 1.0
        self.T = 1.0

    def mcmillian_wavefunction(self, positions):
        """Return wave equation."""
        w = 1.0
        term = 0.0

        for i in range(self.num_p):
            for j in range(i, self.num_p-1):
                # ri_minus_rj = np.subtract(positions[i, :], positions[j+1, :])
                r = 0.0
                for k in range(self.num_d):
                    ri_minus_rj = positions[i, k] - positions[j+1, k]
                    r += ri_minus_rj**2
                distance = math.sqrt(r)
                # distance = math.sqrt(np.sum(np.square(ri_minus_rj)))
                term += (self.k*self.T/distance)**5
        w = math.exp(-0.5*term)

        return w

    def alpha_gradient_wavefunction(self, positions):
        """Calculate derivative of wave function divided by wave function."""
        """This expression holds for the case of the trail wave function
        described by the single particle wave function as a the harmonic
        oscillator function and the correlation function
        """
        gradient = np.zeros((self.num_p, self.num_d))
        for k in range(self.num_p):
            xk = positions[k, 0]
            yk = positions[k, 1]
            zk = positions[k, 2]
            rk = np.array((xk, yk, zk))
            for j in range(self.num_p):
                xj = positions[j, 0]
                yj = positions[j, 1]
                zj = positions[j, 2]
                rj = np.array((xj, yj, zj))
                if(j != k):
                    rkj = math.sqrt(np.sum((rk - rj)*(rk - rj)))
                    gradient[k, :] = 5*0.5*self.k*self.T*(rk-rj)/rkj

        return gradient

    def wavefunction_ratio(self, positions, new_positions):
        """Wave function with new positions squared divided by."""
        """wave equation with old positions squared"""
        wf_old = self.mcmillian_wavefunction(positions)
        wf_new = self.mcmillian_wavefunction(new_positions)
        numerator = wf_new*wf_new
        denominator = wf_old*wf_old
        acceptance_ratio = numerator/denominator

        return acceptance_ratio

    def quantum_force(self, positions):
        """Return drift force."""
        """This surely is inefficient, rewrite so the quantum force matrix
        gets updated, than calculating it over and over again each time"""
        quantum_force = np.zeros((self.num_p, self.num_d))
        position_forward = np.array(positions)
        psi_current = self.mcmillian_wavefunction(positions)
        psi_moved = 0.0
        step = 0.001

        for i in range(self.num_p):
            for j in range(self.num_d):
                position_forward[i, j] = position_forward[i, j] + step
                psi_moved = self.mcmillian_wavefunction(position_forward)
                # Resett positions
                position_forward[i, j] = position_forward[i, j] - step
                derivative = (psi_moved - psi_current)/step
                quantum_force[i, j] = (2.0/psi_current)*derivative

        return quantum_force
