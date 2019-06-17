import numpy as np
import math
import sys


class System:

	#deri_psi = 0.0
	#g        = 0.0
	#f        = 0.0

	def __init__(self, num_particles, num_dimensions,
		alpha, beta, a):

		self.num_p = num_particles
		self.num_d = num_dimensions
		self.alpha = alpha
		self.beta  = beta
		self.a     = a


	def wavefunction(self, positions):

		wf = self.single_particle_function(positions)*self.jastrow_factor(positions)
		return wf


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
				g = g*math.exp(-self.alpha*(x*x + y*y + z*z))

			else:
				g = g*math.exp(-self.alpha*(x*x + y*y))
		#g = np.prod(math.exp(-self.alpha*(np.sum(np.power(positions, 2), axis=1))))
		return g


	def jastrow_factor(self, positions):

		f = 1.0
		n = self.num_d - 1

		for i in range(self.num_p):
			for j in range(i, self.num_p-1):
				distance = abs(np.subtract(positions[i,n], positions[j+1,n]))

				if distance > self.a:
					f *= 1.0 - self.a/distance
				else:
					f *= 0	
		return f


	def derivative_psi_term(self, positions):

		"""
		This expression holds for the case of the trail wave function
		described by the single particle wave function as a the harmonic
		oscillator function and the correlation function
		"""
		deri_psi = 1.0

		for i in range(self.num_p):
			x = positions[i,0]
			y = positions[i,1]
			if self.num_d > 2:
				positions[i,2] *= self.beta #if vector is 3 dimesions
				z = positions[i,2]
				deri_psi *= (x*x + y*y + z*z)
			else:
				deri_psi *= (x*x + y*y)

		return -deri_psi
