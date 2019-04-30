import numpy as np
import sys
import matplotlib.pyplot as plt


from system import System


"""
Variational Monte Carlo with Metropolis Hastings algorithm for selection of 
configurations. Optimizing using Gradient descent.
"""

monte_carlo_cycles     = 5
number_of_particles    = 3
number_of_dimensions   = 3
step_length            = 0.1
variational_parameters = np.array[0.1, 1.0, 0.1, 1.0]
delta_R                = 0.1
delta_t                = 0.1
positions = np.random.rand(number_of_particles, number_of_dimensions)

for i in range(monte_carlo_cycles):

	"""
	Run the metropolis algo for given Monte Carlo cycles.
	"""
	s = systems(number_of_particles, number_of_dimensions, 
		positions, variational_parameters, step_length)

	m = metropolis(s, delta_R. delta_t)

