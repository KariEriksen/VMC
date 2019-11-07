"""Hamiltonian class."""
from hamiltonian import Hamiltonian # noqa: 401


class Lennard_Jones:
    """Calculate variables regarding the Hamiltonian of given wavefunction."""

    def __init__(self, epsilon, sigma, wavefunction):
        """Instance of class."""
        self.epsilon = epsilon
        self.sigma = sigma
        self.s = wavefunction

    def laplacian_numerical(self, positions):
        """Numerical differentiation for solving laplacian."""

        omega = 1.0
        Hamiltonian(omega, self.s.wavefunction)
        laplacian = Hamiltonian.laplacian_numerical(positions)

        return laplacian

    def laplacian_analytical(self, positions):
        """Analytical solution to the laplacian"""

        return 0

    def lennard_jones_potential(self, positions):
        """Regular Lennard-Jones potential"""
        C6 = (self.sigma/positions)**6
        C12 = (self.sigma/positions)**12
        V = 4*self.epsilon*(C12 - C6)

        return V
