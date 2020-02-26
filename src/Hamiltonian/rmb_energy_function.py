"""Sampler class."""
import numpy as np
import math


class RBM_Energy_Function:
    """Calculate variables regarding energy of given system."""

    def __init__(self, gamma, omega, numerical_step, system):
        """Instance of class."""
        self.gamma = gamma
        self.omega = omega
        self.omega2 = omega*omega
        self.step = numerical_step
        self.s = system

    def local_energy(self, positions):
        """Return the local energy."""

        Xi = 0.0
        fd, sd = self.derivative_wavefunction
        interaction = self.interaction
        for i in range(self.s.M):
            Xi += self.positions[i]
        local_energy = 0.5*(-fd*fd + sd + self.omega2*Xi) + interaction

        return local_energy

    def derivatives_wavefunction(self, positions):
        """Return the first and second derivative of ln of the wave function"""

        first_derivative = 0.0
        second_derivative = 0.0

        for i in range(self.s.M):
            sum2 = 0.0
            sum3 = 0.0
            for j in range(self.s.N):
                sum1 = 0.0
                for k in range(self.s.M):
                    sum1 += self.positions[k]*self.s.W[k, j]/self.s.sigma2

                exponent = math.exp(-self.s.b[j] - sum1)
                sum2 += self.s.W[i, j]/(1 + exponent)
                sum3 += sum2*sum2*exponent

            first_derivative += (-(self.positions[i] - self.a[i])/self.s.sigma2
                                 + (1/self.s.sigma2)*sum2)

            second_derivative += -1/self.s.sigma2 + (1/self.s.sigma4)*sum3

        return first_derivative, second_derivative

    def derivatives_qd_wavefunction(self, positions):
        """Return the first and second derivative of ln of the"""
        """quadratic wave function"""
        """Used in Gibbs sampling"""

        first_derivative_gibbs = 0.5*self.derivatives_wavefunction[0]
        second_derivative_gibbs = 0.5*self.derivatives_wavefunction[1]

        return first_derivative_gibbs, second_derivative_gibbs

    def interaction(self, positions):
        """Return the interaction between particles"""

        return 0

    def local_energy_times_wf(self, positions):
        """Return local energy times the derivative of wave equation."""

        energy = self.local_energy(positions)
        energy_times_wf_a = self.s.derivative_wavefunction(positions)*energy
        energy_times_wf_b = self.s.derivative_wavefunction(positions)*energy
        energy_times_wf_W = self.s.derivative_wavefunction(positions)*energy

        return energy_times_wf_a, energy_times_wf_b, energy_times_wf_W

    def probability(self, positions, new_positions):
        """Wave function with new positions squared divided by."""
        """wave equation with old positions squared"""

        wf_old = self.s.wavefunction(positions)
        wf_new = self.s.wavefunction(new_positions)
        numerator = wf_new*wf_new
        denominator = wf_old*wf_old
        acceptance_ratio = numerator/denominator

        return acceptance_ratio
