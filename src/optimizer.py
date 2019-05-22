import numpy as np

from sampler import Sampler
from metropolis import Metropolis
from system import System

class Optimizer:

	def __init__(self, learning_rate, iterations, sampler, metropolis):

		self.learning_rate = learning_rate
		self.iterations    = iterations
		self.s             = sampler
		self.m             = metropolis

	def gradient_descent(self):

		for i in range(self.iterations):

			der_energy = self.s.energy_gradient(posistions)
			new_alpha  = alpha - self.learning_rate*der_energy
			System.update(alpha)

			new_energy = m.metropolis() 

		return 0


