"""Sampler class."""
import numpy as np
import math


class Sampler:
    """Calculate variables regarding energy of given wavefunction."""

    def __init__(self, omega, wavefunction, hamiltonian):
        """Instance of class."""
        self.omega = omega
        self.s = wavefunction
        self.h = hamiltonian

    def local_energy_times_wf(self, positions):
        """Return local energy times the derivative of wave equation."""
        energy = self.local_energy(positions)
        energy_times_wf = self.s.gradient_wavefunction(positions)*energy

        return energy_times_wf
