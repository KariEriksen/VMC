import numpy as np
import sys
import math

class System:

	def __init__(self, num_particles, num_dimensions, positions,
		alpha, beta, a):

		self.num_particles          = num_p
		self.num_dimensionsn        = num_d
		self.alpha                  = alpha
		self.beta                   = beta
		self.a                      = a


	def wavefunction(self, positions):

		return single_particel_function(positions)*jastrow_factor(positions)


	def single_particel_function(self, positions):

		"""
		Takes in position matrix of the particles and calculates the
		single particle wave function. 
		Returns g, type float, product of all single particle wave functions
		of all particles.
		"""

		for i in range(self.num_p):

			positions[i,3] *= self.beta 

		g = np.prod(math.exp(-self.alpha*(np.sum(np.power(positions, 2), axis=1))))

		return g


	def jastrow_factor(self, positions):

		f = 0

		for i in range(self.num_p):
			for j in range(self.num_p-(i+1)):
				j = i + 1
				distance = abs(np.subtract(positions[i,3], positions[j,3]))

				if distance > self.a:
					f *= 1.0 - self.a/distance
				else:
					f *= 0
			
		return f



