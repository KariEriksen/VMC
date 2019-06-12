import numpy as np

from system import System

class Sampler:
	

	#num_particles = 
	#S = System(num_particles, num_dimensions, alpah, beta, a)

	def __init__(self, omega, numerical_step, system):

		self.omega = omega
		self.step  = numerical_step
		self.s     = system


	def kinetic_energy(self, positions):

		"""
		Numerical differentiation for solving the second derivative
		of the wave function. 
		Step represents small changes is the spatial space
		"""

		position_forward  = positions + self.step
		position_backward = positions - self.step

		lambda_ = (self.s.wavefunction(position_forward) 
				+ self.s.wavefunction(position_backward) 
				- 2*self.s.wavefunction(positions))*(1/(self.step*self.step))

		kine_energy = lambda_/self.s.wavefunction(positions)

		return kine_energy


	def potential_energy(self, positions):

		"""
		Returns the potential energy of the system

		np.multiply multiply argument element-wise
		"""

		omega_sq = self.omega*self.omega

		return 0.5*omega_sq*np.multiply(positions, positions)


	def local_energy(self, positions):

		energy =  -0.5*self.kinetic_energy(positions) + self.potential_energy(positions)

		return energy


	def local_energy_times_wf(self, positions):

		energy = self.local_energy(positions)
		energy_times_wf = self.s.derivative_psi_term(positions)*energy

		return energy_times_wf


	def probability(self, positions, new_positions):

		acceptance_ratio = (self.s.wavefunction(new_positions)
						 *self.s.wavefunction(new_positions)
						 /self.s.wavefunction(positions)
						 *self.s.wavefunction(positions))

		return acceptance_ratio


	def drift_force(self, positions):

		position_forward  = positions + self.step
		derivativ = (self.s.wavefunction(position_forward) 
				  - self.s.wavefunction(positions))/self.step
		return derivativ


	def greens_function(self, positions, new_positions_importance):

		greens_function = 0.0

		F_old = self.drift_force(positions)
		F_new = self.drift_force(new_positions_importance)

		greens_function = (0.5*(F_old + F_new)
		                *(0.5*(positions - new_positions_importance))
		                + D*self.delta_t*(F_old - F_new))

		greens_function = exp(greens_function)

		return greens_function

