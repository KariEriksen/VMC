import numpy as np
import math
import sys


class System:

	def __init__(self, num_particles, num_dimensions,
		alpha, beta, a):

		self.num_p = num_particles
		self.num_d = num_dimensions
		self.alpha = alpha
		self.beta  = beta
		self.a     = a


	def wavefunction(self, positions):

		return self.single_particle_function(positions)*self.jastrow_factor(positions)


	def single_particle_function(self, positions):

		"""
		Takes in position matrix of the particles and calculates the
		single particle wave function. 
		Returns g, type float, product of all single particle wave functions
		of all particles.
		"""
		g = 1.0

		for i in range(self.num_p):

			#self.num_d = j
			x = positions[i,0]
			y = positions[i,1]
			if self.num_d > 2:
				positions[i,2] *= self.beta #if vector is 3 dimesions
				z = positions[i,2]
				  
			g *= math.exp(-self.alpha*(x*x + y*y + z*z))

		#g = np.prod(math.exp(-self.alpha*(np.sum(np.power(positions, 2), axis=1))))

		return g


	def jastrow_factor(self, positions):

		f = 1.0

		for i in range(self.num_p):
			for j in range(self.num_p-(i+1)):
				j = i + 1
				distance = abs(np.subtract(positions[i,2], positions[j,2]))

				if distance > self.a:
					f *= 1.0 - self.a/distance
				else:
					f *= 0
					#pass
			
		return f


	def derivative_psi_term(self, positions):

		"""
		This expression holds for the case of the trail wave function 
		described by the single particle wave function as a the harmonic 
		oscillator function and the correlation function 
		"""

		value = 1.0

		for i in range(self.num_p):
			x = positions[i,0]
			y = positions[i,1]
			if self.num_d > 2:
				positions[i,2] *= self.beta #if vector is 3 dimesions
				z = positions[i,2]
			
			value *= -(x*x + y*y + z*z)

		return value
