"""Variational Monte Carlo."""

import numpy as np
import sys

# sys.path.append('/Users/morten/Desktop/VMC-1/src')
sys.path.append('/home/kari/VMC/src')
from metropolis import Metropolis
from optimizer import Optimizer
from sampler import Sampler
from system import System

"""
Variational Monte Carlo with Metropolis Hastings algorithm for selection of
configurations. Optimizing using Gradient descent.
"""

monte_carlo_cycles = 1000
num_particles = 2
num_dimensions = 3
numerical_step_length = 0.001
step_metropolis = 0.01
step_importance = 0.1
alpha = 0.4
beta = 1.0
a = 0.0
omega = 1.0
learning_rate = 0.01
gradient_iterations = 50
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

        new_energy, new_positions, count = met.metropolis(positions)
        positions = new_positions
        accumulate_energy += sam.local_energy(positions)

        accumulate_psi_term += sys.derivative_psi_term(positions)
        accumulate_both += sam.local_energy_times_wf(positions)

    expec_val_energy = accumulate_energy/(monte_carlo_cycles*num_particles)
    expec_val_psi = accumulate_psi_term/(monte_carlo_cycles*num_particles)
    expec_val_both = accumulate_both/(monte_carlo_cycles*num_particles)

    derivative_energy = 2*(expec_val_both - expec_val_psi*expec_val_energy)
    print ('deri energy = ', derivative_energy)
    print ('counter (accepted moves in metropolis) = ', count)
    return derivative_energy, new_energy


for i in range(gradient_iterations):

    d_El, energy = run_vmc(parameter)
    new_parameter = opt.gradient_descent(parameter, d_El)
    parameter = new_parameter
    print ('new alpha =  ', new_parameter)
    print ('----------------------------')
    print ('new energy =  ', energy)
