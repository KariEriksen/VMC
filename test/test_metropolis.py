import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from metropolis import Metropolis  # noqa: 401
from Hamiltonian.hamiltonian import Hamiltonian  # noqa; 401
from wavefunction import Wavefunction  # noqa: 401


def test_metropolis():

    delta_R = 1.0
    delta_t = 0.01
    num_particles = 1
    num_dimensions = 2
    omega = 1.0
    a = 0.0
    alpha = np.random.uniform(1e-3, 10)
    beta = np.random.uniform(1e-3, 10)
    wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
    ham = Hamiltonian(omega, wave)
    met = Metropolis(delta_R, delta_t, num_particles, num_dimensions, ham, 0.0)
    positions = np.zeros(shape=(num_particles, num_dimensions))
    _, new_positions, _ = met.metropolis(positions)

    assert new_positions.shape == (num_particles, num_dimensions)


def test_importance_hampling():

    delta_R = 1.0
    delta_t = 0.01
    num_particles = 1
    num_dimensions = 2
    omega = 1.0
    a = 0.0
    alpha = np.random.uniform(1e-3, 10)
    beta = np.random.uniform(1e-3, 10)
    wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
    ham = Hamiltonian(omega, wave)
    met = Metropolis(delta_R, delta_t, num_particles, num_dimensions, ham, 0.0)
    positions = np.zeros(shape=(num_particles, num_dimensions))
    _, new_positions, _ = met.importance_hampling(positions)

    assert new_positions.shape == (num_particles, num_dimensions)
