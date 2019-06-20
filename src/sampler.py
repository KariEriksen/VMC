"""Sampler class."""
import numpy as np


class Sampler:
    """Docstring."""

    # num_particles =
    # S = System(num_particles, num_dimensions, alpah, beta, a)

    def __init__(self, omega, numerical_step, system):
        """Docstring."""
        self.omega = omega
        self.step = numerical_step
        self.s = system

    def kinetic_energy(self, positions):
        """Numerical differentiation."""
        kine_energy = 0.0

        position_forward = positions
        position_backward = positions
        psi_current = 0.0
        psi_moved = 0.0

        for i in range(self.s.num_p):
            psi_current += 2*self.s.num_d*self.s.wavefunction(positions)
            for j in range(self.s.num_d):

                position_forward[i, j] += self.step
                position_backward[i, j] -= self.step
                wf_p = self.s.wavefunction(position_forward)
                wf_n = self.s.wavefunction(position_backward)
                psi_moved += wf_p + wf_n
                # Resett positions
                position_forward[i, j] = positions[i, j]
                position_backward[i, j] = positions[i, j]

            kine_energy = (psi_moved - psi_current)/(self.step*self.step)
            # kine_energy = kine_energy/self.s.wavefunction(positions)

        return kine_energy

    def potential_energy(self, positions):
        """Return the potential energy of the system."""
        omega_sq = self.omega*self.omega

        return 0.5*omega_sq*np.sum(np.multiply(positions, positions))

    def local_energy(self, positions):
        """Docstring."""
        k = self.kinetic_energy(positions)/self.s.wavefunction(positions)
        p = self.potential_energy(positions)
        energy = -0.5*k + p

        return energy

    def local_energy_times_wf(self, positions):
        """Docstring."""
        energy = self.local_energy(positions)
        energy_times_wf = self.s.derivative_psi_term(positions)*energy

        return energy_times_wf

    def probability(self, positions, new_positions):
        """Docstring."""
        wf_old = self.s.wavefunction(positions)
        wf_new = self.s.wavefunction(new_positions)
        numerator = wf_new*wf_new
        denominator = wf_old*wf_old
        acceptance_ratio = numerator/denominator

        return acceptance_ratio

    def drift_force(self, positions):
        """Docstring."""
        position_forward = positions + self.step
        wf_forward = self.s.wavefunction(position_forward)
        wf_current = self.s.wavefunction(positions)
        derivativ = (wf_forward - wf_current)/self.step

        return derivativ

    def greens_function(self, positions, new_positions_importance):
        """Docstring."""
        greens_function = 0.0
        D = 0.0

        F_old = self.drift_force(positions)
        F_new = self.drift_force(new_positions_importance)

        # Deal with this mess later
        greens_function = (0.5*(F_old + F_new) * (0.5 * (positions -
                           new_positions_importance)) +
                           D*self.delta_t*(F_old - F_new))

        greens_function = np.exp(greens_function)

        return greens_function
