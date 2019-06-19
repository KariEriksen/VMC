import numpy as np
from sampler import Sampler
import random


class Metropolis:

	#Sampler(omega, step)

	def __init__(self, delta_R, delta_t, num_particles, num_dimensions,
				 sampler, c):

		self.delta_R   = delta_R
		self.delta_t   = delta_t
		self.num_p     = num_particles
		self.num_d     = num_dimensions
		#self.positions = positions
		self.s         = sampler
		self.c         = c

	"""
	def new_positions(self):


		Calculating new trial position using old position.
		r is a random variable in [0,1] and delta_R is the step length
		in the spatial configuration space


		r = np.random.rand(num_p, num_d)
		new_positions = self.positions + np.multiply(r, self.delta_R)
		return new_positions
	"""


	def metropolis(self, positions):

		"""
		Running the naive metropolis algorithm with brute-force sampling of
		new positions
		"""

		#new_positions = new_positions()
		#r = np.random.rand(self.num_p, self.num_d)
		r = random.random()
		#Pick a random particle and suggest a new move
		random_index = random.randrange(len(positions))
		new_positions = positions
		new_positions[random_index,:] = new_positions[random_index,:] + r*self.delta_R
		acceptance_ratio = self.s.probability(positions, new_positions)
		epsilon = np.random.sample()

		if acceptance_ratio > epsilon:
			positions = new_positions
			self.c += 1.0

		else:
			pass

		energy = self.s.local_energy(positions)

		return energy, positions, self.c


	def importance_sampling(self, positions):

		"""
		Running Importance sampling with upgraded method for suggesting new
		positions. Given through the Langevin equation.
		D is the diffusion coefficient equal 0.5, xi is a gaussion random variable
		and delta_t is the time step between 0.001 and 0.01
		"""

		D  = 0.5
		xi = np.random.sampler()
		new_positions = (positions + D*F*self.delta_t
								 + xi*sqrt(self.delta_t))

		acceptance_ratio = self.s.greens_function(posistions, new_positions)
		epsilon = np.random.sample()

		if acceptance_ratio <= epsilon:
			positions = new_positions

		else:
			pass

		self.s.local_energy(positions)


	def gibbs_sampling(self):

		"""
		Running Gibbs sampling
		"""
