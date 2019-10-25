import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from Hamiltonian.hamiltonian_testing import Hamiltonian  # noqa: 401
from Hamiltonian.non_interaction import Non_Interaction # noqa: 401
from Hamiltonian.weak_interaction import Weak_Interaction # noqa: 401
from Wavefunction.wavefunction import Wavefunction  # noqa: 401
from sampler import Sampler  # noqa: 401


def test_local_energy_times_wf_2d_2p():

    a = 0.0
    num_particles = 2
    num_dimensions = 2
    positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        ham = Hamiltonian(omega, wave)

        e = ham.local_energy_weak_interaction_numerical(positions)
        t = wave.gradient_wavefunction(positions)*e
        test_ener_times = ham.local_energy_times_wf_weak_interaction(positions)
        assert t == pytest.approx(test_ener_times, abs=1e-14)


def test_local_energy_times_wf_3d():

    a = 0.0
    num_particles = 1
    num_dimensions = 3
    positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions[0, 2] = np.random.uniform(-2, 2)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        ham = Hamiltonian(omega, wave)

        e = ham.local_energy_weak_interaction_numerical(positions)
        t = wave.gradient_wavefunction(positions)*e
        test_ener_times = ham.local_energy_times_wf_weak_interaction(positions)
        assert t == pytest.approx(test_ener_times, abs=1e-14)
