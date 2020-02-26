import numpy as np
import sys
import os
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from metropolis import Metropolis # noqa: 401
from optimizer import Optimizer # noqa: 401
from Hamiltonian.non_interaction import Non_Interaction # noqa: 401
from Hamiltonian.weak_interaction import Weak_Interaction # noqa: 401
from Wavefunction.wavefunction import Wavefunction # noqa: 401
from sampler import Sampler # noqa: 401
from system import System # noqa: 401


def iterations(monte_carlo_cycles, num_particles, num_dimensions):
    """Run the variational monte carlo."""

    step_metropolis = 1.0
    step_importance = 0.01
    learning_rate = 0.01
    gradient_iterations = 100

    runs = 0
    step = 1
    n = int(gradient_iterations/step)
    list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 90, 100]
    n = len(list)

    d_El_array = np.zeros(n)
    energy_array = np.zeros(n)
    var_array = np.zeros(n)
    parameter_array = np.zeros(n)
    iter = np.linspace(step, gradient_iterations, n)
    sys = System(num_particles, num_dimensions)

    a = 0.0
    beta = omega = 1.0
    alpha = 0.1

    for i in range(n):
        parameter = alpha
        # runs += step
        opt = Optimizer(learning_rate)
        for j in range(list[i]):

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
            # new_parameter = opt.gradient_descent(parameter, d_El)
            new_parameter = opt.gradient_descent_barzilai_borwein(parameter,
                                                                  d_El, i)
            # print ('new alpha = ', new_parameter)
            # print ('number of gradien descent runs = ', i)
            parameter = new_parameter

        d_El_array[i] = d_El
        energy_array[i] = energy
        var_array[i] = var
        parameter_array[i] = new_parameter
        print (list[i])

    with open('/home/kari/VMC/data/non_interaction_gradient_descent_BB.csv', 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["iterations", "alpha", "derivative_energy",
                         "local_energy", "variance"])
        for i in range(len(d_El_array)):
            writer.writerow([list[i], parameter_array[i], d_El_array[i],
                            energy_array[i], var_array[i]])


iterations(10000, 1, 3)
