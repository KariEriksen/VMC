"""Variational Monte Carlo."""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from metropolis import Metropolis # noqa: 401
from optimizer import Optimizer # noqa: 401
from sampler import Sampler # noqa: 401
from system import System # noqa: 401

"""
Variational Monte Carlo with Metropolis Hastings algorithm for selection of
configurations. Optimizing using Gradient descent.
"""

monte_carlo_cycles = 10000
num_particles = 2
num_dimensions = 2
numerical_step_length = 0.001
step_metropolis = 1.0
step_importance = 0.01
alpha = 0.4
beta = 1.0
a = 0.0
omega = 1.0
learning_rate = 0.01
gradient_iterations = 1000
parameter = alpha
# energy = 0.0
# parameters = np.zeros(gradient_iterations)

opt = Optimizer(learning_rate)


def run_vmc(parameter):
    """Run the variational monte carlo."""
    # Set all values to zero for each new Monte Carlo run
    accumulate_energy = 0.0
    accumulate_psi_term = 0.0
    accumulate_both = 0.0
    new_energy = 0.0

    # Initialize the posistions for each new Monte Carlo run
    positions = np.random.rand(num_particles, num_dimensions)

    # Call system class in order to set new alpha parameter
    sys = System(num_particles, num_dimensions, parameter, beta, a)
    sam = Sampler(omega, numerical_step_length, sys)
    met = Metropolis(step_metropolis, step_importance, num_particles,
                     num_dimensions, sam, 0.0)
    for i in range(monte_carlo_cycles):

        # new_energy, new_positions, count = met.metropolis(positions)
        new_energy, new_positions, count = met.importance_sampling(positions)
        positions = new_positions
        accumulate_energy += sam.local_energy(positions)

        accumulate_psi_term += sys.derivative_psi_term(positions)
        accumulate_both += sam.local_energy_times_wf(positions)

    expec_val_energy = accumulate_energy/(monte_carlo_cycles)
    expec_val_psi = accumulate_psi_term/(monte_carlo_cycles)
    expec_val_both = accumulate_both/(monte_carlo_cycles)

    derivative_energy = 2*(expec_val_both - expec_val_psi*expec_val_energy)
    print ('deri energy = ', derivative_energy)
    print ('counter (accepted moves in metropolis) = ', count)
    return derivative_energy, expec_val_energy


for i in range(gradient_iterations):

    d_El, energy = run_vmc(parameter)
    new_parameter = opt.gradient_descent(parameter, d_El)
    parameter = new_parameter
    print ('new alpha =  ', new_parameter)
    print ('----------------------------')
    print ('new energy =  ', energy, 'correct energy = ',
           0.5*num_dimensions*num_particles)
