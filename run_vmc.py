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

positions = np.random.rand(num_particles, num_dimensions)

Sys = System(num_particles, num_dimensions, alpha, beta, a)

Sam = Sampler(omega, numerical_step_length)

Opt = Optimizer(learning_rate)

Met = Metropolis(step_metropolis, step_importance, num_particles, 
			   num_dimensions, positions)

for i in range(monte_carlo_cycles):

	"""
	Run the metropolis algo for given Monte Carlo cycles.
	"""
	new_posistion = m.metropolis()
	o.gradient_descent(alpha, new_positions)


