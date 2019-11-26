"""Optimizer class."""
import nupy as np


class Optimizer:
    """Optimization method."""

    """The optimizer method runs through a whole Monte Carlo loop
    for each gradient descent iteration. Update of the variational
    parameter is done within the run_vmc file."""

    def __init__(self, learning_rate):
        """Instance of class."""
        self.learning_rate = learning_rate

    def gradient_descent(self, alpha, derivative_energy):
        """Orinary gradient descent."""
        new_alpha = alpha - self.learning_rate*derivative_energy

        return new_alpha

    def gradient_descent_barzilai_borwein(self, alpha, derivative_energy, k):
        """gradient descent with Barzilai-Borwein update on learning rate"""

        old_derivative_energy = derivative_energy
        old_alpha = alpha

        if k == 0:
            gamma = self.learning_rate
            new_alpha = alpha - gamma*derivative_energy

        else:
            diff_alpha = alpha - old_alpha
            diff_derivative_energy = derivative_energy - old_derivative_energy
            gamma = diff_alpha - 1/diff_derivative_energy
            new_alpha = alpha - gamma*derivative_energy

        return new_alpha
