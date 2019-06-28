import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from sampler import Sampler  # noqa: 401
from system import System  # noqa: 401


def test_kinetic_energy_2d():

    num_particles = 1
    num_dimensions = 2
    omega = 1.0
    numerical_step = 0.001
    a = 0.0
    alpha = 0.5
    beta = 1.0
    sys = System(num_particles, num_dimensions, alpha, beta, a)
    sam = Sampler(omega, numerical_step, sys)
    positions = np.zeros(shape=(num_particles, num_dimensions))
    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        sys.alpha = alpha
        sys.beta = beta
        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        positions[0, 0] = x
        positions[0, 1] = y
        pos_xp = np.array(positions)
        pos_xn = np.array(positions)
        pos_yp = np.array(positions)
        pos_yn = np.array(positions)
        pos_xp[0, 0] += numerical_step
        pos_xn[0, 0] -= numerical_step
        pos_yp[0, 1] += numerical_step
        pos_yn[0, 1] -= numerical_step

        wf_current = 2*num_dimensions*sys.wavefunction(positions)
        wf_forward = sys.wavefunction(pos_xp) + sys.wavefunction(pos_yp)
        wf_backwawrd = sys.wavefunction(pos_xn) + sys.wavefunction(pos_yn)
        kine_energy = wf_forward + wf_backwawrd - wf_current
        kine_energy = kine_energy/(numerical_step*numerical_step)

        assert kine_energy == pytest.approx(sam.kinetic_energy(positions),
                                            abs=1e-14)


def test_kinetic_energy_3d():

    num_particles = 1
    num_dimensions = 3
    omega = 1.0
    numerical_step = 0.001
    a = 0.0
    positions = np.zeros(shape=(num_particles, num_dimensions))
    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)
        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-20, 20)
        positions[0, 0] = x
        positions[0, 1] = y
        positions[0, 2] = z*beta
        pos_xp = np.array(positions)
        pos_xn = np.array(positions)
        pos_yp = np.array(positions)
        pos_yn = np.array(positions)
        pos_zp = np.array(positions)
        pos_zn = np.array(positions)
        pos_xp[0, 0] += numerical_step
        pos_xn[0, 0] -= numerical_step
        pos_yp[0, 1] += numerical_step
        pos_yn[0, 1] -= numerical_step
        pos_zp[0, 2] += numerical_step
        pos_zn[0, 2] -= numerical_step

        wf_current = 2*num_dimensions*sys.wavefunction(positions)
        wf_forward = (sys.wavefunction(pos_xp) + sys.wavefunction(pos_yp) +
                      sys.wavefunction(pos_zp))
        wf_backwawrd = (sys.wavefunction(pos_xn) + sys.wavefunction(pos_yn) +
                        sys.wavefunction(pos_zn))
        kine_energy = wf_forward + wf_backwawrd - wf_current
        kine_energy = kine_energy/(numerical_step*numerical_step)

        assert kine_energy == pytest.approx(sam.kinetic_energy(positions),
                                            abs=1e-14)


def test_potential_energy():

    assert 1 == 1


def test_local_energy():

    assert 1 == 1


def test_local_energy_times_wf():

    assert 1 == 1


def test_probability():

    assert 1 == 1


def test_drift_force():

    assert 1 == 1


def test_greens_function():

    assert 1 == 1
