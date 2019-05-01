import numpy as np
import sys
import math

class System:

	def __init__(self, num_particles, num_dimensions, positions,
		variational_parameters, step_length):

		self.num_particles          = num_p
		self.num_dimensionsn        = num_d
		self.positions              = positions
		self.variational_parameters = vari_p
		self.numerical_step_length  = step

		alpha     = vari_p[0]
		beta      = vari_p[1]
		omega     = vari_p[3]
		a         = vari_p[4]


	def wavefunction(positions):

		return single_particel_function(positions)*jastrow_factor(positions)


	def single_particel_function(positions):

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


	def jastrow_factor(positions):

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



