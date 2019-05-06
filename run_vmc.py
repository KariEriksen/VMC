import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('/home/kari/VMC/src')

from system import System
from sampler import Sampler
from metropolis import Metropolis

"""
Variational Monte Carlo with Metropolis Hastings algorithm for selection of 
configurations. Optimizing using Gradient descent.
"""

monte_carlo_cycles       = 5
number_of_particles      = 3
number_of_dimensions     = 3
numerical_step_length    = 0.1
step_metropolis          = 0.1
step_importance          = 0.1
alpha                    = 0.1
beta                     = 1.0
a                        = 0.0
omega                    = 0.01

positions = np.random.rand(number_of_particles, number_of_dimensions)

System(number_of_particles, number_of_dimensions,
	   alpha, beta, a)

Sampler(omega, numerical_step_length)

m = Metropolis(step_metropolis, step_importance, number_of_particles, 
			   number_of_dimensions, positions)

for i in range(monte_carlo_cycles):

	"""
	Run the metropolis algo for given Monte Carlo cycles.
	"""
	m.metropolis()


