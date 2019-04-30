import numpy as np
import sys
import math

class System:

	def __init__(self, num_particles, num_dimensions, 
		variational_parameters, step_length):

		self.num_particles          = num_p
		self.num_dimensionsn        = num_d
		self.variational_parameters = vari_p
		self.step_length            = step

		positions = np.random.rand(num_p, num_d)
		alpha     = vari_p[0]
		beta      = vari_p[1]
		omega     = vari_p[3]
		a         = vari_p[4]


	def hamiltonian():

		return -0.5*kinetic_energy() + potential_energy()


	def kinetic_energy():

		"""
		Numerical differentiation for solving the second derivative
		of the wave function. 
		Step represents small changes is the spatial space
		"""

		position_forward  = positions + step
		position_backward = positions - step

		lambda_ = (wavefunction(position_forward) 
				+ wavefunction(position_backwards) 
				- 2*wavefunction(positions))*(1/(step*step))

		kine_energy = lambda_/wavefunction()

		return kine_energy


	def potential_energy():

		"""
		Returns the potential energy of the system

		np.multiply multiply argument element-wise
		"""

		omega_sq = omega*omega

		return 0.5*omega_sq*np.multiply(positions, positions)


	def wavefunction():

		return single_particel_function()*jastrow_factor()


	def single_particel_function():

		"""
		Takes in position matrix of the particles and calculates the
		single particle wave function. 
		Returns g, type float, product of all single particle wave functions
		of all particles.
		"""

		for i in range(num_p):

			positions[i,3] *= beta 

		g = np.prod(math.exp(-alpha*(np.sum(np.power(positions, 2), axis=1))))

		return g


	def jastrow_factor():

		f = 0

		for i in range(num_p):
			for j in range(num_p-(i+1)):
				j = i + 1
				distance = abs(np.subtract(positions[i,3], positions[j,3]))

				if distance > a:
					f *= 1.0 - a/distance
				else:
					f *= 0
			
		return f



