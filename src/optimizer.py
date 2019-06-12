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

	def __init__(self, alpha, learning_rate, derivative_energy):

		self.alpha              = alpha
		self.learning_rate      = learning_rate
		self.derivative_energy  = derivative_energy

	def gradient_descent(self):

		new_alpha  = self.alpha - self.learning_rate*self.derivative_energy

		return new_alpha

