import numpy as np

from system import System

class Sampler:

	#num_particles = 
	#S = System(num_particles, num_dimensions, alpah, beta, a)

	def __init__(self, omega, step):

		self.omega = omega
		self.step  = step


	def kinetic_energy(self, positions, s):

		"""
		Numerical differentiation for solving the second derivative
		of the wave function. 
		Step represents small changes is the spatial space
		"""

		position_forward  = positions + self.step
		position_backward = positions - self.step

		lambda_ = (s.wavefunction(position_forward) 
				+ System.wavefunction(position_backwards) 
				- 2*System.wavefunction(positions))*(1/(self.step*self.step))

		kine_energy = lambda_/System.wavefunction(positions)

		return kine_energy


	def potential_energy(self, positions):

		"""
		Returns the potential energy of the system

		np.multiply multiply argument element-wise
		"""

		omega_sq = self.omega*self.omega

		return 0.5*omega_sq*np.multiply(positions, positions)


	def local_energy(self, positions):

		return -0.5*kinetic_energy(positions) + potential_energy(positions)


	def energy_gradient(self, positions):

		return 0


	def probability(self, positions, new_positions):

		acceptance_ratio = (System.wavefunction(new_positions)
						 *System.wavefunction(new_positions)
						 /System.wavefunction(positions)
						 *System.wavefunction(positions))

		return acceptance_ratio


	def drift_force(self, positions):

		position_forward  = positions + self.step
		derivativ = (System.wavefunction(position_forward) 
				  - System.wavefunction(positions))/self.step
		return derivativ


	def greens_function(self, positions, new_positions_importance):

		greens_function = 0.0

		F_old = drift_force(positions)
		F_new = drift_force(new_positions_importance)

		greens_function = (0.5*(F_old + F_new)
		                *(0.5*(positions - new_positions_importance))
		                + D*self.delta_t*(F_old - F_new))

		greens_function = exp(greens_function)

		return greens_function
