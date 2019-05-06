import numpy as np
import sys

from sampler import Sampler

class Metropolis:

	#Sampler(omega, step)

	def __init__(self, delta_R, delta_t, num_particles, num_dimensionsn,
				 positions, step_metropolis):

		self.delta_R                = delta_R
		self.delta_t                = delta_t
		self.num_particles          = num_p
		self.num_dimensionsn        = num_d
		self.positions              = positions

	"""
	def new_positions(self):

		
		Calculating new trial position using old position.
		r is a random variable in [0,1] and delta_R is the step length 
		in the spatial configuration space
		

		r = np.random.rand(num_p, num_d)
		new_positions = self.positions + np.multiply(r, self.delta_R)
		return new_positions
	"""


	def metropolis(self):

		"""
		Running the naive metropolis algorithm with brute-force sampling of
		new positions
		"""

		#new_positions = new_positions()
		r = np.random.rand(self.num_p, self.num_d)
		new_positions = self.positions + np.multiply(r, self.delta_R)
		acceptance_ratio = Sampler.probability(self.positions, new_positions)
		epsilon = np.random.sampler()

		if acceptance_ratio < epsilon:
			self.positions = new_positions

		else:
			pass

		Sampler.local_energy(self.positions)


	def importance_sampling(self):
	
		"""
		Running Importance sampling with upgraded method for suggesting new
		positions. Given through the Langevin equation. 
		D is the diffusion coefficient equal 0.5, xi is a gaussion random variable
		and delta_t is the time step between 0.001 and 0.01
		"""

		D  = 0.5
		xi = np.random.sampler()
		new_positions_importance = self.positions + D*F*self.delta_t 
									  + xi*sqrt(self.delta_t)
		
		acceptance_ratio = Sampler.greens_function(self.posistions, new_positions_importance)
		epsilon = np.random.sampler()

		if acceptance_ratio < epsilon:
			self.positions = new_positions_importance

		else:
			pass

		Sampler.local_energy(self.positions)


	def gibbs_sampling(self):
	
		"""
		Running Gibbs sampling 
		"""	 

	




