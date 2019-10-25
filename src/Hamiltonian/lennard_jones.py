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

    def lennard_jones_potential(self, positions):
        """Regular Lennard-Jones potential"""
        C6 = (self.sigma/positions)**6
        C12 = (self.sigma/positions)**12
        V = 4*self.epsilon*(C12 - C6)

        return V
