"""Variational Monte Carlo."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from metropolis import Metropolis # noqa: 401
from optimizer import Optimizer # noqa: 401
from Hamiltonian.non_interaction import Non_Interaction # noqa: 401
from Hamiltonian.weak_interaction import Weak_Interaction # noqa: 401
from Wavefunction.wavefunction import Wavefunction # noqa: 401
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
# Hamiltonian.update(self, alpha)


def non_interaction_case(monte_carlo_cycles, num_particles, num_dimensions,
                         alpha):
    """Run the variational monte carlo."""

    a = 0.0
    beta = omega = 1.0
    if alpha is None:
        alpha = 0.49

    parameter = alpha
    for i in range(gradient_iterations):

        # Call wavefunction class in order to set new alpha parameter
        wave = Wavefunction(num_particles, num_dimensions, parameter, beta, a)
        # Run with analytical expression of local energy = true
        hamilton = Non_Interaction(omega, wave, 'true')
        met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                         num_particles, num_dimensions, wave, hamilton)

        # d_El = met.run_metropolis()
        d_El = met.run_importance_sampling('true')
        new_parameter = opt.gradient_descent(parameter, d_El)
        # new_parameter = opt.gradient_descent_barzilai_borwein(parameter,
        #                                                       d_El, i)
        print ('new alpha = ', new_parameter)
        print ('number of gradien descent runs = ', i)
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

        # Call wavefunction class in order to set new alpha parameter
        wave = Wavefunction(num_particles, num_dimensions, parameter, beta, a)
        # Run with analytical expression of local energy = true
        hamilton = Weak_Interaction(omega, wave, 'true')
        met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                         num_particles, num_dimensions, wave, hamilton)

        # d_El = met.run_metropolis()
        # Run with analytical expression for quantum force = true
        d_El = met.run_importance_sampling('true')
        new_parameter = opt.gradient_descent(parameter, d_El)
        # new_parameter = opt.gradient_descent_barzilai_borwein(parameter,
        #                                                      d_El, i)
        print ('new alpha = ', new_parameter)
        print ('number of gradien descent runs = ', i)
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

        # Call wavefunction class in order to set new alpha parameter
        wave = Wavefunction(num_particles, num_dimensions, parameter, beta, a)
        # Run with analytical expression of local energy = true
        hamilton = Weak_Interaction(omega, wave, 'false')
        met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                         num_particles, num_dimensions, wave, hamilton)

        d_El = met.run_metropolis()
        # d_El = met.run_importance_sampling('false')
        new_parameter = opt.gradient_descent(parameter, d_El)
        print ('new alpha = ', new_parameter)
        print ('number of gradien descent runs = ', i)
        parameter = new_parameter


def brute_force(monte_carlo_cycles, num_particles, num_dimensions, alpha):
    """Run the variational monte carlo"""
    """using brute force"""

    a = 0.0
    beta = omega = 1.0
    alpha_start = 0.1
    alpha_stop = 1.0
    alpha_step = 0.02
    n = int((alpha_stop - alpha_start)/alpha_step)

    parameter = alpha_start
    for i in range(n):

        # Call wavefunction class in order to set new alpha parameter
        wave = Wavefunction(num_particles, num_dimensions, parameter, beta, a)
        # Run with analytical expression of local energy = true
        hamilton = Non_Interaction(omega, wave, 'true')
        met = Metropolis(monte_carlo_cycles, step_metropolis, step_importance,
                         num_particles, num_dimensions, wave, hamilton)

        # d_El = met.run_metropolis()
        d_El = met.run_importance_sampling('true')
        # new_parameter = opt.gradient_descent_barzilai_borwein(parameter,
        #                                                       d_El, i)

        parameter += alpha_step


"""case(monte_carlo_cycles, number of particles,
        number of dimensions, interaction parameter)"""

# non_interaction_case(100000, 2, 3, 0.48)
# weak_interaction_case(100000, 2, 3, 0.47)
# elliptic_weak_interaction_case(10000, 2, 3, None)
brute_force(100000, 2, 3, None)
