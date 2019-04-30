import numpy as np
import sys


from system import System

class Sampler(System):

	"""

	"""

	def local_energy():


	def energy_gradient():


	def probability():


	def drift_force():


	def greens_function():

		greens_function = 0.0

		F_old = Sampler.drift_force(positions)
		F_new = Sampler.drift_force(new_positions_importance)

		greens_function = 0.5*(F_old + F_new)
		                *(0.5*(positions - new_positions_importance) 
		                + D*delta_t*(F_old - F_new))

		greens_function = exp(greens_function)

		return greens_function
