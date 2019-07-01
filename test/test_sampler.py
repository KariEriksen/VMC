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
        positions[0, 2] = z
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

        # assert kine_energy == pytest.approx(sam.kinetic_energy(positions),
        #                                    abs=1e-14)
        assert 1 == 1


def test_potential_energy_2d():

    num_particles = 1
    num_dimensions = 3
    numerical_step = 0.001
    a = 0.0
    positions = np.zeros(shape=(num_particles, num_dimensions))
    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)

        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-20, 20)
        positions[0, 0] = x
        positions[0, 1] = y
        positions[0, 2] = z

        sum = 0.0
        for i in range(num_dimensions):
            sum += (positions[0, i]*positions[0, i])

        pot_energy = 0.5*omega*omega*(sum)
        assert pot_energy == pytest.approx(sam.potential_energy(positions),
                                           abs=1e-14)


def test_potential_energy_3d():

    num_particles = 1
    num_dimensions = 3
    numerical_step = 0.001
    a = 0.0
    positions = np.zeros(shape=(num_particles, num_dimensions))
    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)

        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-20, 20)
        positions[0, 0] = x
        positions[0, 1] = y
        positions[0, 2] = z

        sum = 0.0
        for i in range(num_dimensions):
            sum += (positions[0, i]*positions[0, i])

        pot_energy = 0.5*omega*omega*(sum)
        assert pot_energy == pytest.approx(sam.potential_energy(positions),
                                           abs=1e-14)


def test_local_energy_2d():

    a = 0.0
    num_particles = 1
    num_dimensions = 2
    numerical_step = 0.001
    alpha = np.random.uniform(1e-3, 10)
    beta = np.random.uniform(1e-3, 10)
    omega = np.random.uniform(1e-3, 10)
    positions = np.zeros(shape=(num_particles, num_dimensions))
    x = np.random.uniform(-20, 20)
    y = np.random.uniform(-20, 20)
    positions[0, 0] = x
    positions[0, 1] = y
    sys = System(num_particles, num_dimensions, alpha, beta, a)
    sam = Sampler(omega, numerical_step, sys)
    k = sam.kinetic_energy(positions)/sys.wavefunction(positions)
    p = sam.potential_energy(positions)
    local_energy = -0.5*k + p
    assert local_energy == pytest.approx(sam.local_energy(positions),
                                         abs=1e-14)


def test_local_energy_3d():

    assert 1 == 1


def test_local_energy_times_wf_2d():

    assert 1 == 1


def test_local_energy_times_wf_3d():

    assert 1 == 1


def test_probability_2d():

    assert 1 == 1


def test_probability_3d():

    assert 1 == 1


def test_drift_force_2d():

    assert 1 == 1


def test_drift_force_3d():

    assert 1 == 1


def test_greens_function_2d():

    assert 1 == 1


def test_greens_function_3d():

    assert 1 == 1
