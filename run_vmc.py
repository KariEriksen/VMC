import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('/home/kari/VMC/src')

from system     import System
from sampler    import Sampler
from metropolis import Metropolis
from optimizer  import Optimizer

"""
Variational Monte Carlo with Metropolis Hastings algorithm for selection of 
configurations. Optimizing using Gradient descent.
"""

monte_carlo_cycles       = 5
num_particles            = 3
num_dimensions           = 3
numerical_step_length    = 0.1
step_metropolis          = 0.1
step_importance          = 0.1
alpha                    = 0.1
beta                     = 1.0
a                        = 0.0
omega                    = 0.01
learning_rate            = 0.01
gradient_iterations      = 2

positions = np.random.rand(num_particles, num_dimensions)

Sys = System(num_particles, num_dimensions, alpha, beta, a)
Sam = Sampler(omega, numerical_step_length, Sys)
Met = Metropolis(step_metropolis, step_importance, num_particles, 
			   num_dimensions, positions, Sam)
Opt = Optimizer(learning_rate, gradient_iterations, Sam, Met)


def run_vmc(parameters):

	#Set all values to zero for each new Monte Carlo run
	accumulate_energy   = 0.0
	accumulate_psi_term = 0.0
	accumulate_both     = 0.0

	for i in range(self.monte_carlo_cycles):

		new_energy = Met.metropolis() 
		accumulate_energy   += Sam.local_energy(...) 
		accumulate_psi_term += Sys.derivative_psi_term(...)
		accumulate_both     += Sam.local_energy_times_wf()

	expec_value_energy = accumulate_energy/(monte_carlo_cycles*n_particles)
	expec_value_psi    = accumulate_psi_term/(monte_carlo_cycles*n_particles)
	expec_value_both   = accumulate_both/(monte_carlo_cycles*n_particles)

	derivative_energy = 2*(expec_value_both - expec_value_psi*expec_value_energy)

	return derivative_energy


for i in range(gradient_iterations):
	d_El = run_vmc(paramters)
	dparameters = Opt.gradient_descent()
	paramters += dparameters





