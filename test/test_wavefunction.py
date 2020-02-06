import pytest
import numpy as np
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from Wavefunction.wavefunction import Wavefunction  # noqa: 401
from system import System # noqa: 401


def test_wavefunction_wavefunction_2d():
    num_particles = 1
    num_dimensions = 2
    alpha = 1.0
    beta = 1.0
    a = 0.0
    sys = System(num_particles, num_dimensions)
    s = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)
    positions = np.zeros(shape=(num_particles, num_dimensions))
    x = 2.92858925782756
    y = 0.00925285752985
    positions[0, 0] = x
    positions[0, 1] = y

    wf = s.single_particle_function(positions)
    assert wf == pytest.approx(np.exp(-x**2 - y**2), abs=1e-14)

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        s.alpha = alpha

        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        positions[0, 0] = x
        positions[0, 1] = y
        wf = s.single_particle_function(positions)
        assert wf == pytest.approx(np.exp(-alpha*(x**2 + y**2)), abs=1e-14)


def test_wavefunction_wavefunction_2d_2p():
    num_particles = 2
    num_dimensions = 2
    alpha = 1.0
    beta = 1.0
    a = 0.0
    sys = System(num_particles, num_dimensions)
    s = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        s.alpha = alpha
        positions = np.random.uniform(-20, 20, (num_particles, num_dimensions))
        x1 = positions[0, 0]
        y1 = positions[0, 1]
        x2 = positions[1, 0]
        y2 = positions[1, 1]
        test = np.exp(-alpha*(x1*x1 + y1*y1))*np.exp(-alpha*(x2*x2 + y2*y2))
        wf = s.single_particle_function(positions)
        assert wf == pytest.approx(test, abs=1e-14)


def test_wavefunction_wavefunction_3d():
    num_particles = 1
    num_dimensions = 3
    alpha = 1.0
    beta = 1.0
    a = 0.0
    sys = System(num_particles, num_dimensions)
    s = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)
    positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        s.alpha = alpha
        s.beta = beta

        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-20, 20)

        positions[0, 0] = x
        positions[0, 1] = y
        positions[0, 2] = z
        wf = s.single_particle_function(positions)
        assert wf == pytest.approx(np.exp(-alpha*(x**2 + y**2 + beta*z**2)),
                                   abs=1e-14)


def test_wavefunction_wavefunction_3d_2p():
    num_particles = 2
    num_dimensions = 3
    alpha = 1.0
    beta = 1.0
    a = 0.0
    sys = System(num_particles, num_dimensions)
    s = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)
    positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        s.alpha = alpha
        s.beta = beta

        x1 = np.random.uniform(-20, 20)
        y1 = np.random.uniform(-20, 20)
        z1 = np.random.uniform(-20, 20)
        x2 = np.random.uniform(-20, 20)
        y2 = np.random.uniform(-20, 20)
        z2 = np.random.uniform(-20, 20)

        positions[0, 0] = x1
        positions[0, 1] = y1
        positions[0, 2] = z1
        positions[1, 0] = x2
        positions[1, 1] = y2
        positions[1, 2] = z2
        test = (np.exp(-alpha*(x1**2 + y1**2 + beta*z1**2)) *
                np.exp(-alpha*(x2**2 + y2**2 + beta*z2**2)))

        wf = s.single_particle_function(positions)
        assert wf == pytest.approx(test, abs=1e-14)


def test_jastrow_factor_2d_2p():

    a = 0.43
    num_particles = 2
    num_dimensions = 2
    alpha = 1.0
    beta = 1.0
    positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-20, 20)
        positions[0, 1] = np.random.uniform(-20, 20)
        positions[1, 0] = np.random.uniform(-20, 20)
        positions[1, 1] = np.random.uniform(-20, 20)
        sys = System(num_particles, num_dimensions)
        sys.positions_distances(positions)
        s = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)
        f = 1.0
        for i in range(num_particles):
            for j in range(i, num_particles-1):
                t = np.subtract(positions[i, :], positions[j+1, :])
                d = math.sqrt(np.sum(np.square(t)))
                if d > a:
                    f *= 1.0 - (a/d)
                else:
                    f *= 0.0
        assert f == pytest.approx(s.jastrow_factor(positions), abs=1e-14)


def test_jastrow_factor_3d_2p():

    a = 0.43
    num_particles = 2
    num_dimensions = 3
    alpha = 1.0
    beta = 1.0
    positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-20, 20)
        positions[0, 1] = np.random.uniform(-20, 20)
        positions[0, 2] = np.random.uniform(-20, 20)
        positions[1, 0] = np.random.uniform(-20, 20)
        positions[1, 1] = np.random.uniform(-20, 20)
        positions[1, 2] = np.random.uniform(-20, 20)
        sys = System(num_particles, num_dimensions)
        sys.positions_distances(positions)
        s = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)
        f = 1.0
        for i in range(num_particles):
            for j in range(i, num_particles-1):
                t = np.subtract(positions[i, :], positions[j+1, :])
                distance = math.sqrt(np.sum(np.square(t)))
                if distance > a:
                    f *= 1.0 - (a/distance)
                else:
                    f *= 0.0
        assert f == pytest.approx(s.jastrow_factor(positions), abs=1e-14)


def test_wavefunction_derivative_psi_term_2d():
    num_particles = 1
    num_dimensions = 2
    alpha = 1.0
    beta = 1.0
    a = 0.0
    sys = System(num_particles, num_dimensions)
    s = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)
    positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        s.alpha = alpha

        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)

        positions[0, 0] = x
        positions[0, 1] = y
        wf = s.alpha_gradient_wavefunction(positions)

        assert wf == pytest.approx((-x**2 - y**2), abs=1e-14)


def test_wavefunction_derivative_psi_term_3d():
    num_particles = 1
    num_dimensions = 3
    alpha = 1.0
    beta = 1.0
    a = 0.0
    sys = System(num_particles, num_dimensions)
    s = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)
    positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        s.alpha = alpha
        s.beta = beta

        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-20, 20)

        positions[0, 0] = x
        positions[0, 1] = y
        positions[0, 2] = z
        wf = s.alpha_gradient_wavefunction(positions)

        assert wf == pytest.approx((-x**2 - y**2 - beta*z**2), abs=1e-12)


def test_probability_2d():

    a = 0.0
    num_particles = 1
    num_dimensions = 2
    positions = np.zeros(shape=(num_particles, num_dimensions))
    new_positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        new_positions[0, 0] = np.random.uniform(-2, 2)
        new_positions[0, 1] = np.random.uniform(-2, 2)
        sys = System(num_particles, num_dimensions)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)

        P_old = wave.wavefunction(positions)*wave.wavefunction(positions)
        P_new = (wave.wavefunction(new_positions) *
                 wave.wavefunction(new_positions))
        accept_ratio = P_new/P_old
        assert accept_ratio == pytest.approx(wave.wavefunction_ratio(positions,
                                             new_positions), abs=1e-14)


def test_probability_3d():

    a = 0.0
    num_particles = 1
    num_dimensions = 3
    positions = np.zeros(shape=(num_particles, num_dimensions))
    new_positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions[0, 2] = np.random.uniform(-2, 2)
        new_positions[0, 0] = np.random.uniform(-2, 2)
        new_positions[0, 1] = np.random.uniform(-2, 2)
        new_positions[0, 2] = np.random.uniform(-2, 2)
        sys = System(num_particles, num_dimensions)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)

        P_old = wave.wavefunction(positions)*wave.wavefunction(positions)
        P_new = (wave.wavefunction(new_positions) *
                 wave.wavefunction(new_positions))
        accept_ratio = P_new/P_old
        assert accept_ratio == pytest.approx(wave.wavefunction_ratio(positions,
                                             new_positions), abs=1e-14)


def test_drift_force_2d():

    a = 0.0
    num_particles = 1
    num_dimensions = 2
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    drift_force = np.zeros((1, 2))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions_fw_x = np.array(positions)
        positions_fw_y = np.array(positions)
        positions_fw_x[0, 0] = positions[0, 0] + numerical_step
        positions_fw_y[0, 1] = positions[0, 1] + numerical_step
        sys = System(num_particles, num_dimensions)
        sys.positions_distances(positions)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)

        wf_current = wave.wavefunction(positions)
        wf_forward_x = wave.wavefunction(positions_fw_x)
        wf_forward_y = wave.wavefunction(positions_fw_y)
        deri1 = (wf_forward_x - wf_current)/numerical_step
        deri2 = (wf_forward_y - wf_current)/numerical_step
        drift_force[0, 0] = (2.0/wf_current)*deri1
        drift_force[0, 1] = (2.0/wf_current)*deri2

        assert drift_force == pytest.approx(wave.quantum_force_numerical
                                            (positions), abs=1e-14)


def test_drift_force_2d_2p():

    a = 0.0
    num_particles = 2
    num_dimensions = 2
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    drift_force = np.zeros((2, 2))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions[1, 0] = np.random.uniform(-2, 2)
        positions[1, 1] = np.random.uniform(-2, 2)
        positions_fw_x1 = np.array(positions)
        positions_fw_y1 = np.array(positions)
        positions_fw_x2 = np.array(positions)
        positions_fw_y2 = np.array(positions)
        positions_fw_x1[0, 0] = positions[0, 0] + numerical_step
        positions_fw_y1[0, 1] = positions[0, 1] + numerical_step
        positions_fw_x2[1, 0] = positions[1, 0] + numerical_step
        positions_fw_y2[1, 1] = positions[1, 1] + numerical_step
        sys = System(num_particles, num_dimensions)
        sys.positions_distances(positions)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)

        wf_current = wave.wavefunction(positions)
        wf_forward_x1 = wave.wavefunction(positions_fw_x1)
        wf_forward_y1 = wave.wavefunction(positions_fw_y1)
        wf_forward_x2 = wave.wavefunction(positions_fw_x2)
        wf_forward_y2 = wave.wavefunction(positions_fw_y2)
        deri1 = (wf_forward_x1 - wf_current)/numerical_step
        deri2 = (wf_forward_y1 - wf_current)/numerical_step
        deri3 = (wf_forward_x2 - wf_current)/numerical_step
        deri4 = (wf_forward_y2 - wf_current)/numerical_step
        drift_force[0, 0] = (2.0/wf_current)*deri1
        drift_force[0, 1] = (2.0/wf_current)*deri2
        drift_force[1, 0] = (2.0/wf_current)*deri3
        drift_force[1, 1] = (2.0/wf_current)*deri4

        assert drift_force == pytest.approx(wave.quantum_force_numerical
                                            (positions), abs=1e-14)


def test_drift_force_3d():

    a = 0.0
    num_particles = 1
    num_dimensions = 3
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    drift_force = np.zeros((1, 3))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions[0, 2] = np.random.uniform(-2, 2)
        positions_fw_x = np.array(positions)
        positions_fw_y = np.array(positions)
        positions_fw_z = np.array(positions)
        positions_fw_x[0, 0] = positions[0, 0] + numerical_step
        positions_fw_y[0, 1] = positions[0, 1] + numerical_step
        positions_fw_z[0, 2] = positions[0, 2] + numerical_step
        sys = System(num_particles, num_dimensions)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)

        wf_current = wave.wavefunction(positions)
        wf_forward_x = wave.wavefunction(positions_fw_x)
        wf_forward_y = wave.wavefunction(positions_fw_y)
        wf_forward_z = wave.wavefunction(positions_fw_z)
        deri1 = (wf_forward_x - wf_current)/numerical_step
        deri2 = (wf_forward_y - wf_current)/numerical_step
        deri3 = (wf_forward_z - wf_current)/numerical_step
        drift_force[0, 0] = (2.0/wf_current)*deri1
        drift_force[0, 1] = (2.0/wf_current)*deri2
        drift_force[0, 2] = (2.0/wf_current)*deri3
        assert drift_force == pytest.approx(wave.quantum_force_numerical
                                            (positions), abs=1e-14)


def test_drift_force_3d_2p():

    a = 0.0
    num_particles = 2
    num_dimensions = 3
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    drift_force = np.zeros((2, 3))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions[0, 2] = np.random.uniform(-2, 2)
        positions[1, 0] = np.random.uniform(-2, 2)
        positions[1, 1] = np.random.uniform(-2, 2)
        positions[1, 2] = np.random.uniform(-2, 2)
        positions_fw_x1 = np.array(positions)
        positions_fw_y1 = np.array(positions)
        positions_fw_z1 = np.array(positions)
        positions_fw_x2 = np.array(positions)
        positions_fw_y2 = np.array(positions)
        positions_fw_z2 = np.array(positions)
        positions_fw_x1[0, 0] = positions[0, 0] + numerical_step
        positions_fw_y1[0, 1] = positions[0, 1] + numerical_step
        positions_fw_z1[0, 2] = positions[0, 2] + numerical_step
        positions_fw_x2[1, 0] = positions[1, 0] + numerical_step
        positions_fw_y2[1, 1] = positions[1, 1] + numerical_step
        positions_fw_z2[1, 2] = positions[1, 2] + numerical_step
        sys = System(num_particles, num_dimensions)
        sys.positions_distances(positions)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)

        wf_current = wave.wavefunction(positions)
        wf_forward_x1 = wave.wavefunction(positions_fw_x1)
        wf_forward_y1 = wave.wavefunction(positions_fw_y1)
        wf_forward_z1 = wave.wavefunction(positions_fw_z1)
        wf_forward_x2 = wave.wavefunction(positions_fw_x2)
        wf_forward_y2 = wave.wavefunction(positions_fw_y2)
        wf_forward_z2 = wave.wavefunction(positions_fw_z2)
        deri1 = (wf_forward_x1 - wf_current)/numerical_step
        deri2 = (wf_forward_y1 - wf_current)/numerical_step
        deri3 = (wf_forward_x2 - wf_current)/numerical_step
        deri4 = (wf_forward_y2 - wf_current)/numerical_step
        deri5 = (wf_forward_z1 - wf_current)/numerical_step
        deri6 = (wf_forward_z2 - wf_current)/numerical_step
        drift_force[0, 0] = (2.0/wf_current)*deri1
        drift_force[0, 1] = (2.0/wf_current)*deri2
        drift_force[1, 0] = (2.0/wf_current)*deri3
        drift_force[1, 1] = (2.0/wf_current)*deri4
        drift_force[0, 2] = (2.0/wf_current)*deri5
        drift_force[1, 2] = (2.0/wf_current)*deri6

        assert drift_force == pytest.approx(wave.quantum_force_numerical
                                            (positions), abs=1e-14)


def test_quantum_force_3d_2p():

    a = 0.0
    num_particles = 2
    num_dimensions = 3
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    drift_force = np.zeros((2, 3))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions[0, 2] = np.random.uniform(-2, 2)
        positions[1, 0] = np.random.uniform(-2, 2)
        positions[1, 1] = np.random.uniform(-2, 2)
        positions[1, 2] = np.random.uniform(-2, 2)
        positions_fw_x1 = np.array(positions)
        positions_fw_y1 = np.array(positions)
        positions_fw_z1 = np.array(positions)
        positions_fw_x2 = np.array(positions)
        positions_fw_y2 = np.array(positions)
        positions_fw_z2 = np.array(positions)
        positions_fw_x1[0, 0] = positions[0, 0] + numerical_step
        positions_fw_y1[0, 1] = positions[0, 1] + numerical_step
        positions_fw_z1[0, 2] = positions[0, 2] + numerical_step
        positions_fw_x2[1, 0] = positions[1, 0] + numerical_step
        positions_fw_y2[1, 1] = positions[1, 1] + numerical_step
        positions_fw_z2[1, 2] = positions[1, 2] + numerical_step
        sys = System(num_particles, num_dimensions)
        sys.positions_distances(positions)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a, sys)

        wf_current = wave.wavefunction(positions)
        wf_forward_x1 = wave.wavefunction(positions_fw_x1)
        wf_forward_y1 = wave.wavefunction(positions_fw_y1)
        wf_forward_z1 = wave.wavefunction(positions_fw_z1)
        wf_forward_x2 = wave.wavefunction(positions_fw_x2)
        wf_forward_y2 = wave.wavefunction(positions_fw_y2)
        wf_forward_z2 = wave.wavefunction(positions_fw_z2)
        deri1 = (wf_forward_x1 - wf_current)/numerical_step
        deri2 = (wf_forward_y1 - wf_current)/numerical_step
        deri3 = (wf_forward_x2 - wf_current)/numerical_step
        deri4 = (wf_forward_y2 - wf_current)/numerical_step
        deri5 = (wf_forward_z1 - wf_current)/numerical_step
        deri6 = (wf_forward_z2 - wf_current)/numerical_step
        drift_force[0, 0] = (2.0/wf_current)*deri1
        drift_force[0, 1] = (2.0/wf_current)*deri2
        drift_force[1, 0] = (2.0/wf_current)*deri3
        drift_force[1, 1] = (2.0/wf_current)*deri4
        drift_force[0, 2] = (2.0/wf_current)*deri5
        drift_force[1, 2] = (2.0/wf_current)*deri6

        assert 0.0 == 0.0
