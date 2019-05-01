import numpy as np
import sys


from system import System

class Sampler():

	"""

	"""

	def kinetic_energy():

		"""
		Numerical differentiation for solving the second derivative
		of the wave function. 
		Step represents small changes is the spatial space
		"""

		position_forward  = positions + step
		position_backward = positions - step

		lambda_ = (System.wavefunction(position_forward) 
				+ System.wavefunction(position_backwards) 
				- 2*System.wavefunction(positions))*(1/(step*step))

		kine_energy = lambda_/System.wavefunction(positions)

		return kine_energy


	def potential_energy():

		"""
		Returns the potential energy of the system

		np.multiply multiply argument element-wise
		"""

		omega_sq = omega*omega

		return 0.5*omega_sq*np.multiply(positions, positions)

	def local_energy():

		return -0.5*kinetic_energy() + potential_energy()


	def energy_gradient():

		return 0


	def probability(positions, new_positions):

		acceptance_ratio = System.wavefunction(new_positions)*System.wavefunction(new_positions)/
						   System.wavefunction(positions)*System.wavefunction(positions)


	def drift_force(positions):

		position_forward  = positions + step
		derivativ = (System.wavefunction(position_forward) 
				  - System.wavefunction(positions))/step
		return derivativ


	def greens_function(positions, new_positions_importance):

		greens_function = 0.0

		F_old = drift_force(positions)
		F_new = drift_force(new_positions_importance)

		greens_function = 0.5*(F_old + F_new)
		                *(0.5*(positions - new_positions_importance) 
		                + D*delta_t*(F_old - F_new))

		greens_function = exp(greens_function)

		return greens_function
