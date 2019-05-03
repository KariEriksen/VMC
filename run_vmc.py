import numpy as np
import sys
import matplotlib.pyplot as plt


from system import System


"""
Variational Monte Carlo with Metropolis Hastings algorithm for selection of 
configurations. Optimizing using Gradient descent.
"""

monte_carlo_cycles       = 5
number_of_particles      = 3
number_of_dimensions     = 3
metropolis_step_length   = 0.1
numerical_step_length    = 0.1
alpha                    = 0.1
beta                     = 1.0
a                        = 0.0
omega                    = 0.01
position_step_metropolis = 0.1
time_step_importance     = 0.1
positions = np.random.rand(number_of_particles, number_of_dimensions)

for i in range(monte_carlo_cycles):

	"""
	Run the metropolis algo for given Monte Carlo cycles.
	"""

	m = metropolis(delta_R. delta_t)

