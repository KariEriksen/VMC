import numpy as np
import sys


from system import System

class Sampler():

	"""

	"""

	def local_energy():

		return -0.5*kinetic_energy() + potential_energy()


	def energy_gradient():

		return 0


	def probability(positions, new_positions):

		acceptance_ratio = wavefunction(new_positions)*wavefunction(new_positions)/
						   wavefunction(positions)*wavefunction(positions)


	def drift_force():

		position_forward  = positions + step
		derivativ = (wavefunction(position_forward) - wavefunction(positions))/step
		return derivativ


	def greens_function():

		greens_function = 0.0

		F_old = drift_force(positions)
		F_new = drift_force(new_positions_importance)

		greens_function = 0.5*(F_old + F_new)
		                *(0.5*(positions - new_positions_importance) 
		                + D*delta_t*(F_old - F_new))

		greens_function = exp(greens_function)

		return greens_function
