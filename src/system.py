import numpy as np
import sys
import math

class System:

	def __init__(self, num_particles, num_dimensions, positions,
		numerical_step_length, alpha, beta, omega, a):

		self.num_particles          = num_p
		self.num_dimensionsn        = num_d
		self.positions              = positions
		self.numerical_step_length  = step
		self.alpha                  = alpha
		self.beta                   = beta
		self.omega                  = omega
		self.a                      = a


	def wavefunction(self):

		return single_particel_function(self.positions)*jastrow_factor(self.positions)


	def single_particel_function(self):

		"""
		Takes in position matrix of the particles and calculates the
		single particle wave function. 
		Returns g, type float, product of all single particle wave functions
		of all particles.
		"""

		for i in range(self.num_p):

			self.positions[i,3] *= self.beta 

		g = np.prod(math.exp(-self.alpha*(np.sum(np.power(self.positions, 2), axis=1))))

		return g


	def jastrow_factor(self):

		f = 0

		for i in range(self.num_p):
			for j in range(self.num_p-(i+1)):
				j = i + 1
				distance = abs(np.subtract(self.positions[i,3], self.positions[j,3]))

				if distance > self.a:
					f *= 1.0 - self.a/distance
				else:
					f *= 0
			
		return f



