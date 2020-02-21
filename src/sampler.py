"""Sampler class."""


class Sampler:
    """Calculate variables regarding energy of given wavefunction."""

    def __init__(self, wavefunction, hamiltonian):
        """Instance of class."""
        self.w = wavefunction
        self.h = hamiltonian

    def sample_values(self, positions):
        """Sample important values"""
        """From Hamiltonian and Wavefunction class"""

        self.local_energy = self.h.local_energy(positions)
        self.alpha_gradient_wf = self.w.alpha_gradient_wavefunction(positions)
        self.accumulate_energy += self.local_energy
        self.accumulate_energy_sq += self.local_energy*self.local_energy
        self.accumulate_psi_term += self.w.alpha_gradient_wavefunction(positions)
        self.accumulate_both += self.local_energy*self.alpha_gradient_wf

    def average_values(self, monte_carlo_cycles):

        self.expec_val_energy = self.accumulate_energy/monte_carlo_cycles
        self.expec_val_psi = self.accumulate_psi_term/monte_carlo_cycles
        self.expec_val_both = self.accumulate_both/monte_carlo_cycles

        # calculation of variance for the non-interacting case
        # where the exact solution of the expec energy is known
        expec_energy_sq = self.expec_val_energy*self.expec_val_energy
        energy_sq = self.accumulate_energy_sq/monte_carlo_cycles

        self.variance = energy_sq - expec_energy_sq

        self.derivative_energy = 2*(self.expec_val_both -
                                    self.expec_val_psi*self.expec_val_energy)

    def initialize(self):
        """Set all sampling values to zero"""

        self.local_energy = 0.0
        self.alpha_gradient_wf = 0.0
        self.accumulate_energy = 0.0
        self.accumulate_energy_sq = 0.0
        self.accumulate_psi_term = 0.0
        self.accumulate_both = 0.0
        self.expec_val_energy = 0.0
        self.expec_val_psi = 0.0
        self.expec_val_both = 0.0
        self.derivative_energy = 0.0
        self.variance = 0.0
