import numpy as np

from sampler import Sampler
from metropolis import Metropolis
from system import System

class Optimizer:

	"""
	The optimizer method runs through a whole Monte Carlo loop
	for each gradient descent iteration. Update of the variational 
	parameter is done within the run_vmc file.
	"""

	def __init__(self, learning_rate, monte_carlo_cycles, sampler, metropolis):

		self.learning_rate      = learning_rate
		self.monte_carlo_cycles = iterations
		self.s                  = sampler
		self.m                  = metropolis

	def gradient_descent(self):

		for i in range(self.monte_carlo_cycles):

			der_energy = self.s.energy_gradient(posistions)
			new_alpha  = alpha - self.learning_rate*der_energy
			System.update(alpha)

			new_energy = m.metropolis() 

		return 0


