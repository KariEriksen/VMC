import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#sys.path.append('/Users/morten/Desktop/VMC-1/src')
sys.path.append('/home/kari/VMC/src')

from system     import System
from sampler    import Sampler
from metropolis import Metropolis
from optimizer  import Optimizer

"""
Variational Monte Carlo with Metropolis Hastings algorithm for selection of
configurations. Optimizing using Gradient descent.
"""

monte_carlo_cycles       = 1000
num_particles            = 2
num_dimensions           = 2
numerical_step_length    = 0.1
step_metropolis          = 0.1
step_importance          = 0.1
alpha                    = 0.5
beta                     = 1.0
a                        = 0.0
omega                    = 0.01
learning_rate            = 0.01
gradient_iterations      = 20
parameter                = alpha
#energy                   = 0.0
#parameters               = np.zeros(gradient_iterations)

Opt = Optimizer(learning_rate)

def run_vmc(parameter):

	#Set all values to zero for each new Monte Carlo run
	accumulate_energy   = 0.0
	accumulate_psi_term = 0.0
	accumulate_both     = 0.0

	#Initialize the posistions for each new Monte Carlo run
	positions = np.random.rand(num_particles, num_dimensions)

	#Call system class in order to set new alpha parameter
	Sys = System(num_particles, num_dimensions, parameter, beta, a)
	Sam = Sampler(omega, numerical_step_length, Sys)
	Met = Metropolis(step_metropolis, step_importance, num_particles,
			   num_dimensions, Sam)

	for i in range(monte_carlo_cycles):

		new_energy, new_position = Met.metropolis(positions)
		accumulate_energy        += Sam.local_energy(new_position)

		accumulate_psi_term      += Sys.derivative_psi_term(new_position)
		accumulate_both          += Sam.local_energy_times_wf(new_position)

	expec_value_energy = accumulate_energy/(monte_carlo_cycles*num_particles)
	expec_value_psi    = accumulate_psi_term/(monte_carlo_cycles*num_particles)
	expec_value_both   = accumulate_both/(monte_carlo_cycles*num_particles)

	derivative_energy = 2*(expec_value_both - expec_value_psi*expec_value_energy)

	return derivative_energy, new_energy


for i in range(gradient_iterations):

	d_El, energy = run_vmc(parameter)
	new_parameter = Opt.gradient_descent(parameter, d_El)
	parameter = new_parameter

	print (energy)
	#print (parameter)
