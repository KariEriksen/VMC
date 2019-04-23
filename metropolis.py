import numpy as np
import sys

class Metropolis:

	def __init__(self, positions):

		self.positions = positions
		

	def new_positions(self, r):

		"""
		Calculating new trial position using old position.
		r is a random variable in [0,1] and delta_R is the step length 
		in the spatial configuration space
		"""

		new_positions = positions + r*delta_R


	def metropolis(self):

		"""
		Running the naive metropolis algorithm with brute-force sampling of
		new positions
		"""


	def importance_sampling(self):
	
		"""
		Running Importance sampling with upgraded method for suggesting new
		positions. Given through the Langevin equation. 
		D is the diffusion coefficient equal 0.5, xi is a gaussion random variable
		and delta_t is the time step between 0.001 and 0.01
		"""

	def gibbs_sampling(self):
	
		"""
		Running Gibbs sampling 
		"""	 

	




