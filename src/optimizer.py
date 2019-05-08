import numpy as np

from sampler import Sampler

class Optimizer:

	#S = Sampler(omega, step)

	def __init__(self, learning_rate):

		self.learning_rate = learning_rate

	def gradient_descent(self, alpha, posistions):

		der_energy = Sampler.energy_gradient(posistions)
		new_alpha  = alpha - self.learning_rate*der_energy

		return new_alpha


