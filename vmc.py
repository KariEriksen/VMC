"""Variational Monte Carlo."""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from metropolis import Metropolis # noqa: 401
from optimizer import Optimizer # noqa: 401
from Hamiltonian.hamiltonian import Hamiltonian # noqa: 401
from Hamiltonian.non_interaction import Non_Interaction # noqa: 401
from wavefunction import Wavefunction # noqa: 401

"""
Variational Monte Carlo with Metropolis Hastings algorithm for selection of
configurations. Optimizing using Gradient descent.
"""

monte_carlo_cycles = 10000
num_particles = 2
num_dimensions = 3
step_metropolis = 1.0
step_importance = 0.01
learning_rate = 0.01
gradient_iterations = 1000

opt = Optimizer(learning_rate)


def non_interaction_case():
    """Run the variational monte carlo."""

    a = 0.0
    beta = omega = 1.0
    alpha = 0.49
    # Initialize the posistions for each new Monte Carlo run
    positions = np.random.rand(num_particles, num_dimensions)

    # Call wavefunction class in order to set new alpha parameter
    wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
    # Run with numerical expression of local energy = true
    hamilton = Non_Interaction(omega, wave, 'true')
    met = Metropolis(step_metropolis, step_importance, num_particles,
                     num_dimensions, hamilton, 0.0)
    met.metropolis(positions)
    # met.importance_sampling(positions)


def weak_interaction_case():
    """Run the variational monte carlo."""

    a = 0.00433
    beta = omega = 1.0
    alpha = 0.48
    # Initialize the posistions for each new Monte Carlo run
    positions = np.random.rand(num_particles, num_dimensions)

    # Call wavefunction class in order to set new alpha parameter
    wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
    # Run with numerical expression of local energy = true
    ham = Hamiltonian(omega, wave, 'true')
    met = Metropolis(step_metropolis, step_importance, num_particles,
                     num_dimensions, ham, 0.0)
    met.metropolis(positions)
    # met.importance_sampling(positions)


def elliptic_weak_interaction_case():
    """Run the variational monte carlo."""

    a = 0.00433
    beta = omega = 2.82843
    alpha = 0.46
    # Initialize the posistions for each new Monte Carlo run
    positions = np.random.rand(num_particles, num_dimensions)

    # Call wavefunction class in order to set new alpha parameter
    wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
    # Run with numerical expression of local energy = true
    ham = Hamiltonian(omega, wave, 'true')
    met = Metropolis(step_metropolis, step_importance, num_particles,
                     num_dimensions, ham, 0.0)
    met.metropolis(positions)
    # met.importance_sampling(positions)


for i in range(gradient_iterations):

    non_interaction_case()
    # weak_interaction_case()
    # elliptic_weak_interaction_case()
