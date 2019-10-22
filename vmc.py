"""Variational Monte Carlo."""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from metropolis import Metropolis # noqa: 401
from optimizer import Optimizer # noqa: 401
from Hamiltonian.non_interaction import Non_Interaction # noqa: 401
from Hamiltonian.weak_interaction import Weak_Interaction # noqa: 401
from wavefunction import Wavefunction # noqa: 401
from sampler import Sampler # noqa: 401

"""
Variational Monte Carlo with Metropolis Hastings algorithm for selection of
configurations. Optimizing using Gradient descent.
"""
step_metropolis = 1.0
step_importance = 0.01
learning_rate = 0.01
gradient_iterations = 1000

opt = Optimizer(learning_rate)


def non_interaction_case(monte_carlo_cycles, num_particles, num_dimensions,
                         alpha):
    """Run the variational monte carlo."""

    a = 0.0
    beta = omega = 1.0
    if alpha is None:
        alpha = 0.49

    parameter = alpha
    for i in range(gradient_iterations):

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(num_particles, num_dimensions)

        # Call wavefunction class in order to set new alpha parameter
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        # Run with numerical expression of local energy = true
        hamilton = Non_Interaction(omega, wave, 'true')
        met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                         num_particles, num_dimensions, wave, hamilton)

        d_El = met.run_metropolis(positions)
        # d_El = met.run_importance_sampling(positions)
        new_parameter = opt.gradient_descent(parameter, d_El)
        parameter = new_parameter


def weak_interaction_case(monte_carlo_cycles, num_particles, num_dimensions,
                          alpha):
    """Run the variational monte carlo."""

    a = 0.00433
    beta = omega = 1.0
    if alpha is None:
        alpha = 0.48

    parameter = alpha
    for i in range(gradient_iterations):

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(num_particles, num_dimensions)

        # Call wavefunction class in order to set new alpha parameter
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        # Run with numerical expression of local energy = true
        ham = Weak_Interaction(omega, wave, 'true')
        met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                         num_particles, num_dimensions, ham)

        d_El = met.run_metropolis(positions)
        # d_El = met.run_importance_sampling(positions)
        new_parameter = opt.gradient_descent(parameter, d_El)
        parameter = new_parameter


def elliptic_weak_interaction_case(monte_carlo_cycles, num_particles,
                                   num_dimensions, alpha):
    """Run the variational monte carlo."""

    a = 0.00433
    beta = omega = 2.82843
    if alpha is None:
        alpha = 0.46

    parameter = alpha
    for i in range(gradient_iterations):

        # Initialize the posistions for each new Monte Carlo run
        positions = np.random.rand(num_particles, num_dimensions)

        # Call wavefunction class in order to set new alpha parameter
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        # Run with numerical expression of local energy = true
        ham = Weak_Interaction(omega, wave, 'true')
        met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                         num_particles, num_dimensions, ham)

        d_El = met.run_metropolis(positions)
        # d_El = met.run_importance_sampling(positions)
        new_parameter = opt.gradient_descent(parameter, d_El)
        parameter = new_parameter


non_interaction_case(10000, 2, 3, None)
# weak_interaction_case()
# elliptic_weak_interaction_case()
