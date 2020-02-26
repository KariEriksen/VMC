"""Optimizer class."""


class Optimizer:
    """Optimization method."""

    """The optimizer method runs through a whole Monte Carlo loop
    for each gradient descent iteration. Update of the variational
    parameter is done within the run_vmc file."""

    def __init__(self, learning_rate):
        """Instance of class."""
        self.learning_rate = learning_rate
        self.old_alpha = 0.0
        self.old_gradient = 0.0

    def gradient_descent(self, alpha, derivative_energy):
        """Orinary gradient descent."""

        new_alpha = alpha - self.learning_rate*derivative_energy
        if new_alpha <= 0:
            new_alpha = 0.01

        return new_alpha

    def gradient_descent_barzilai_borwein(self, alpha, derivative_energy, k):
        """gradient descent with Barzilai-Borwein update on learning rate"""

        if k == 0:
            new_alpha = alpha - self.learning_rate*derivative_energy
            self.old_alpha = alpha
            self.old_gradient = derivative_energy

        else:

            diff_alpha = alpha - self.old_alpha
            diff_derivative_energy = derivative_energy - self.old_gradient
            new_gamma = diff_alpha/diff_derivative_energy

            new_alpha = alpha - new_gamma*derivative_energy

        self.old_alpha = alpha
        self.old_gradient = derivative_energy

        return new_alpha
