"""Variational Monte Carlo."""

import numpy as np
import sys
import os
import csv
import timeit

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from metropolis import Metropolis # noqa: 401
from optimizer import Optimizer # noqa: 401
from Hamiltonian.non_interaction import Non_Interaction # noqa: 401
from Hamiltonian.weak_interaction import Weak_Interaction # noqa: 401
from Hamiltonian.lennard_jones import Lennard_Jones # noqa: 401
from Wavefunction.wavefunction import Wavefunction # noqa: 401
from Wavefunction.mcmillian import McMillian_Wavefunction # noqa: 401
from sampler import Sampler # noqa: 401
from system import System # noqa: 401

"""
Variational Monte Carlo with Metropolis Hastings algorithm for selection of
configurations. Optimizing using Gradient descent.
"""
step_metropolis = 1.0
step_importance = 0.01
learning_rate = 0.05
gradient_iterations = 500

opt = Optimizer(learning_rate)
# Hamiltonian.update(self, alpha)


def non_interaction_case(monte_carlo_cycles, num_particles, num_dimensions,
                         alpha):
    """Run the variational monte carlo."""

    a = 0.0
    beta = omega = 1.0
    if alpha is None:
        alpha = 0.49

    # d_El_array = np.empty(0)
    # parameter_array = np.empty(0)
    d_El_array = np.zeros(gradient_iterations)
    energy_array = np.zeros(gradient_iterations)
    var_array = np.zeros(gradient_iterations)
    parameter_array = np.zeros(gradient_iterations)

    parameter = alpha
    d_El = 1.0
    start = timeit.default_timer()
    sys = System(num_particles, num_dimensions)
    for i in range(gradient_iterations):

        if abs(d_El) > 1e-20:

            # Call wavefunction class in order to set new alpha parameter
            wave = Wavefunction(num_particles, num_dimensions, parameter,
                                beta, a, sys)
            # Run with analytical expression of local energy = true
            hamilton = Non_Interaction(omega, wave, sys, 'true')
            met = Metropolis(monte_carlo_cycles, step_metropolis,
                             step_importance, num_particles, num_dimensions,
                             wave, hamilton, sys)

            d_El, energy, var = met.run_metropolis()
            # d_El, energy, var = met.run_importance_sampling('true')
            stop = timeit.default_timer()
            new_parameter = opt.gradient_descent(parameter, d_El)
            # new_parameter = opt.gradient_descent_barzilai_borwein(parameter,
            #                                                       d_El, i)
            print ('new alpha = ', new_parameter)
            print ('number of gradien descent runs = ', i)
            print ('run time: ', stop - start)

            d_El_array[i] = d_El
            energy_array[i] = energy
            var_array[i] = var
            parameter_array[i] = new_parameter
            parameter = new_parameter

        else:
            break

    print ('final run time: ', stop - start)
    with open('/home/kari/VMC/data/non_interaction_case_data.csv', 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["alpha", "derivative_energy", "local_energy",
                        "variance"])
        for i in range(len(d_El_array)):
            if parameter_array[i] != 0:
                writer.writerow([parameter_array[i], d_El_array[i],
                                energy_array[i], var_array[i]])


def weak_interaction_case(monte_carlo_cycles, num_particles, num_dimensions,
                          alpha):
    """Run the variational monte carlo."""

    a = 0.00433
    beta = omega = 1.0
    if alpha is None:
        alpha = 0.48

    d_El_array = np.zeros(gradient_iterations)
    energy_array = np.zeros(gradient_iterations)
    parameter_array = np.zeros(gradient_iterations)
    var_array = np.zeros(gradient_iterations)

    parameter = alpha
    start = timeit.default_timer()
    sys = System(num_particles, num_dimensions)
    for i in range(gradient_iterations):

        # Call wavefunction class in order to set new alpha parameter
        wave = Wavefunction(num_particles, num_dimensions, parameter,
                            beta, a, sys)
        # Run with analytical expression of local energy = true
        hamilton = Weak_Interaction(omega, wave, sys, 'true')
        met = Metropolis(monte_carlo_cycles, step_metropolis,
                         step_importance, num_particles, num_dimensions,
                         wave, hamilton, sys)

        d_El, energy, var = met.run_metropolis()
        # Run with analytical expression for quantum force = true
        # d_El, energy, var = met.run_importance_sampling('true')
        stop = timeit.default_timer()
        new_parameter = opt.gradient_descent(parameter, d_El)
        # new_parameter = opt.gradient_descent_barzilai_borwein(parameter,
        #                                                       d_El, i)
        print ('new alpha = ', new_parameter)
        print ('number of gradien descent runs = ', i)
        print ('run time: ', stop - start)

        d_El_array[i] = d_El
        energy_array[i] = energy
        var_array[i] = var
        parameter_array[i] = new_parameter
        parameter = new_parameter

    print ('final run time: ', stop - start)
    with open('/home/kari/VMC/data/weak_interaction_case_data.csv', 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["alpha", "derivative_energy", "local_energy",
                        "variance"])
        for i in range(len(d_El_array)):
            writer.writerow([parameter_array[i], d_El_array[i],
                            energy_array[i], var_array[i]])


def strong_interaction_case(monte_carlo_cycles, num_particles, num_dimensions,
                            alpha):
    """Run the variational monte carlo."""

    if alpha is None:
        alpha = 0.48

    epsilon = 1.0
    sigma = 1.0

    d_El_array = np.zeros(gradient_iterations)
    energy_array = np.zeros(gradient_iterations)
    parameter_array = np.zeros(gradient_iterations)
    var_array = np.zeros(gradient_iterations)

    parameter = alpha
    start = timeit.default_timer()
    sys = System(num_particles, num_dimensions)
    for i in range(gradient_iterations):

        # Call wavefunction class in order to set new alpha parameter
        wave = McMillian_Wavefunction(num_particles, num_dimensions, parameter,
                                      sys)
        # Run with analytical expression of local energy = true
        hamilton = Lennard_Jones(epsilon, sigma, wave, sys, 'true')
        met = Metropolis(monte_carlo_cycles, step_metropolis,
                         step_importance, num_particles, num_dimensions,
                         wave, hamilton, sys)

        d_El, energy, var = met.run_metropolis()
        # Run with analytical expression for quantum force = true
        # d_El, energy, var = met.run_importance_sampling('true')
        stop = timeit.default_timer()
        new_parameter = opt.gradient_descent(parameter, d_El)
        # new_parameter = opt.gradient_descent_barzilai_borwein(parameter,
        #                                                       d_El, i)
        print ('new alpha = ', new_parameter)
        print ('number of gradien descent runs = ', i)
        print ('run time: ', stop - start)

        d_El_array[i] = d_El
        energy_array[i] = energy
        var_array[i] = var
        parameter_array[i] = new_parameter
        parameter = new_parameter

    print ('final run time: ', stop - start)
    with open('/home/kari/VMC/data/strong_interaction_case_data.csv', 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["alpha", "derivative_energy", "local_energy",
                        "variance"])
        for i in range(len(d_El_array)):
            writer.writerow([parameter_array[i], d_El_array[i],
                            energy_array[i], var_array[i]])


def elliptic_weak_interaction_case(monte_carlo_cycles, num_particles,
                                   num_dimensions, alpha):
    """Run the variational monte carlo."""

    a = 0.00433
    beta = omega = 2.82843
    if alpha is None:
        alpha = 0.46

    d_El_array = np.zeros(gradient_iterations)
    energy_array = np.zeros(gradient_iterations)
    parameter_array = np.zeros(gradient_iterations)

    parameter = alpha
    sys = System(num_particles, num_dimensions)
    for i in range(gradient_iterations):

        # Call wavefunction class in order to set new alpha parameter
        wave = Wavefunction(num_particles, num_dimensions, parameter,
                            beta, a, sys)
        # Run with analytical expression of local energy = true
        hamilton = Non_Interaction(omega, wave, sys, 'true')
        met = Metropolis(monte_carlo_cycles, step_metropolis,
                         step_importance, num_particles, num_dimensions,
                         wave, hamilton, sys)

        d_El, energy = met.run_metropolis()
        # d_El, energy = met.run_importance_sampling('false')
        new_parameter = opt.gradient_descent(parameter, d_El)
        print ('new alpha = ', new_parameter)
        print ('number of gradien descent runs = ', i)

        d_El_array[i] = d_El
        energy_array[i] = energy
        parameter_array[i] = new_parameter
        parameter = new_parameter

    with open('/home/kari/VMC/data/non_interaction_case_data.csv', 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["alpha", "derivative_energy", "local_energy"])
        for i in range(len(d_El_array)):
            writer.writerow([parameter_array[i], d_El_array[i],
                            energy_array[i]])


def brute_force(monte_carlo_cycles, num_particles, num_dimensions, alpha):
    """Run the variational monte carlo"""
    """using brute force"""

    a = 0.0
    beta = omega = 1.0
    alpha_start = 0.1
    alpha_stop = 1.0
    alpha_step = 0.02
    n = int((alpha_stop - alpha_start)/alpha_step)

    d_El_array = np.zeros(n)
    energy_array = np.zeros(n)
    parameter_array = np.zeros(n)
    var_array = np.zeros(n)
    parameter = alpha_start
    sys = System(num_particles, num_dimensions)
    for i in range(n):

        # Call wavefunction class in order to set new alpha parameter
        wave = Wavefunction(num_particles, num_dimensions, parameter,
                            beta, a, sys)
        # Run with analytical expression of local energy = true
        hamilton = Non_Interaction(omega, wave, sys, 'true')
        met = Metropolis(monte_carlo_cycles, step_metropolis,
                         step_importance, num_particles, num_dimensions,
                         wave, hamilton, sys)

        d_El, energy, var = met.run_metropolis()
        # d_El, energy, var = met.run_importance_sampling('true')
        # new_parameter = opt.gradient_descent_barzilai_borwein(parameter,
        #                                                       d_El, i)

        var_array[i] = var
        d_El_array[i] = d_El
        energy_array[i] = energy
        parameter_array[i] = parameter
        parameter += alpha_step

    with open('/home/kari/VMC/data/non_interaction_brute_force.csv', 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["alpha", "derivative_energy", "local_energy",
                        "variance"])
        for i in range(len(d_El_array)):
            writer.writerow([parameter_array[i], d_El_array[i],
                            energy_array[i], var_array[i]])


def one_body_density(monte_carlo_cycles, num_particles, num_dimensions, alpha):
    """Run the variational monte carlo"""
    """using brute force"""

    a = 0.0
    beta = omega = 1.0
    alpha = 0.5

    sys = System(num_particles, num_dimensions)
    # Call wavefunction class in order to set new alpha parameter
    wave = Wavefunction(num_particles, num_dimensions, alpha,
                        beta, a, sys)
    # Run with analytical expression of local energy = true
    hamilton = Non_Interaction(omega, wave, sys, 'true')
    met = Metropolis(monte_carlo_cycles, step_metropolis,
                     step_importance, num_particles, num_dimensions,
                     wave, hamilton, sys)

    r_vec = np.linspace(0, 4, 41)
    p_r = met.run_one_body_sampling()
    with open('/home/kari/VMC/data/obd_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["r", "density"])
        for i in range(len(r_vec)):
            writer.writerow([r_vec[i], p_r[i]/monte_carlo_cycles])


def run_blocking(monte_carlo_cycles, num_particles, num_dimensions,
                 alpha):
    """Run the sampling in metropolis to be used for blocking."""

    a = 0.00433
    beta = omega = 1.0
    if alpha is None:
        alpha = 0.495

    sys = System(num_particles, num_dimensions)
    # Call wavefunction class in order to set new alpha parameter
    wave = Wavefunction(num_particles, num_dimensions, alpha,
                        beta, a, sys)
    # Run with analytical expression of local energy = true
    hamilton = Non_Interaction(omega, wave, sys, 'true')
    met = Metropolis(monte_carlo_cycles, step_metropolis,
                     step_importance, num_particles, num_dimensions,
                     wave, hamilton, sys)

    # d_El, energy = met.run_metropolis()
    # Run with analytical expression for quantum force = true
    energy = met.blocking('true')

    with open('/home/kari/VMC/data/blocking.csv', 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["local_energy"])
        for i in range(len(energy)):
            writer.writerow([energy[i]])
