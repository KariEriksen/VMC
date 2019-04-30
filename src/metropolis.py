import numpy as np
import sys


from system import System
from sampler import Sampler

class Metropolis(System):

	def __init__(self, delta_R, delta_t):

		self.delta_R = delta_R

		System.__init__(self, num_particles, num_dimensions,
		variational_parameters, step_length)

		self.num_particles          = num_p
		self.num_dimensionsn        = num_d
		self.variational_parameters = vari_p
		self.step_length            = step

		alpha     = vari_p[0]
		beta      = vari_p[1]
		omega     = vari_p[3]
		a         = vari_p[4]


	def new_positions(self):

		"""
		Calculating new trial position using old position.
		r is a random variable in [0,1] and delta_R is the step length 
		in the spatial configuration space
		"""

		r = np.random.rand(num_p, num_d)
		new_positions = positions + np.multiply(r, delta_R)


	def metropolis(self):

		"""
		Running the naive metropolis algorithm with brute-force sampling of
		new positions
		"""

		acceptance_ratio = Sampler.probability(positions ,new_positions)
		epsilon = np.random.sampler()

		if acceptance_ratio < epsilon:
			positions = new_positions

		else:
			pass

		Sampler.local_energy(positions)


	def importance_sampling(self):
	
		"""
		Running Importance sampling with upgraded method for suggesting new
		positions. Given through the Langevin equation. 
		D is the diffusion coefficient equal 0.5, xi is a gaussion random variable
		and delta_t is the time step between 0.001 and 0.01
		"""

		D  = 0.5
		xi = np.random.sampler()
		new_positions_importance = positions + D*F*delta_t + xi*sqrt(delta_t)
		
		acceptance_ratio = Sampler.greens_function(positions, new_positions_importance)
		epsilon = np.random.sampler()

		if acceptance_ratio < epsilon:
			positions = new_positions_importance

		else:
			pass

		Sampler.local_energy(positions)


	def gibbs_sampling(self):
	
		"""
		Running Gibbs sampling 
		"""	 

	




