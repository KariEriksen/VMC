import numpy as np
import sys


from system import System

class Sampler:

	def __init__(self, positions, omega, step):

		self.positions                = positions
		self.omega                    = omega
		self.step                     = step


	def kinetic_energy(self):

		"""
		Numerical differentiation for solving the second derivative
		of the wave function. 
		Step represents small changes is the spatial space
		"""

		position_forward  = self.positions + self.step
		position_backward = self.positions - self.step

		lambda_ = (System.wavefunction(position_forward) 
				+ System.wavefunction(position_backwards) 
				- 2*System.wavefunction(self.positions))*(1/(self.step*self.step))

		kine_energy = lambda_/System.wavefunction(self.positions)

		return kine_energy


	def potential_energy(self):

		"""
		Returns the potential energy of the system

		np.multiply multiply argument element-wise
		"""

		omega_sq = self.omega*self.omega

		return 0.5*omega_sq*np.multiply(self.positions, self.positions)


	def local_energy(self):

		return -0.5*kinetic_energy() + potential_energy()


	def energy_gradient(self):

		return 0


	def probability(self, new_positions):

		acceptance_ratio = System.wavefunction(self.new_positions)
						 *System.wavefunction(self.new_positions)
						 /System.wavefunction(self.positions)
						 *System.wavefunction(self.positions)

		return acceptance_ratio


	def drift_force(self):

		position_forward  = self.positions + self.step
		derivativ = (System.wavefunction(position_forward) 
				  - System.wavefunction(self.positions))/self.step
		return derivativ


	def greens_function(self, new_positions_importance):

		greens_function = 0.0

		F_old = drift_force(self.positions)
		F_new = drift_force(self.new_positions_importance)

		greens_function = 0.5*(F_old + F_new)
		                *(0.5*(self.positions - self.new_positions_importance) 
		                + D*self.delta_t*(F_old - F_new))

		greens_function = exp(greens_function)

		return greens_function
