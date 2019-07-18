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
    for _ in range(50):
        positions = np.random.uniform(-20, 20, (num_particles, num_dimensions))
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        sys.alpha = alpha
        sys.beta = beta
        kine_energy = 0.0
        pos_fw = np.array(positions)
        pos_bw = np.array(positions)
        for i in range(num_particles):
            for j in range(num_dimensions):
                pos_fw[i, j] = pos_fw[i, j] + numerical_step
                pos_bw[i, j] = pos_bw[i, j] - numerical_step
                wf_current = 2*sys.wavefunction(positions)
                wf_forward = sys.wavefunction(pos_fw)
                wf_backwawrd = sys.wavefunction(pos_bw)
                pos_fw[i, j] = pos_fw[i, j] - numerical_step
                pos_bw[i, j] = pos_bw[i, j] + numerical_step
                kine_energy += wf_forward + wf_backwawrd - wf_current
        kine_energy = kine_energy/(numerical_step*numerical_step)

        assert kine_energy == pytest.approx(sam.kinetic_energy(positions),
                                            abs=1e-10)


def test_kinetic_energy_2d_2p():

    num_particles = 2
    num_dimensions = 2
    omega = 1.0
    numerical_step = 0.001
    a = 0.0
    alpha = 0.5
    beta = 1.0
    sys = System(num_particles, num_dimensions, alpha, beta, a)
    sam = Sampler(omega, numerical_step, sys)
    for _ in range(50):
        positions = np.random.uniform(-20, 20, (num_particles, num_dimensions))
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        sys.alpha = alpha
        sys.beta = beta
        kine_energy = 0.0
        pos_fw = np.array(positions)
        pos_bw = np.array(positions)
        for i in range(num_particles):
            for j in range(num_dimensions):
                pos_fw[i, j] = pos_fw[i, j] + numerical_step
                pos_bw[i, j] = pos_bw[i, j] - numerical_step
                wf_current = 2*sys.wavefunction(positions)
                wf_forward = sys.wavefunction(pos_fw)
                wf_backwawrd = sys.wavefunction(pos_bw)
                pos_fw[i, j] = pos_fw[i, j] - numerical_step
                pos_bw[i, j] = pos_bw[i, j] + numerical_step
                kine_energy += wf_forward + wf_backwawrd - wf_current
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

        assert kine_energy == pytest.approx(sam.kinetic_energy(positions),
                                            abs=1e-10)


def test_kinetic_energy_3d_2p():

    num_particles = 2
    num_dimensions = 3
    omega = 1.0
    numerical_step = 0.001
    a = 0.0
    alpha = 0.5
    beta = 1.0
    sys = System(num_particles, num_dimensions, alpha, beta, a)
    sam = Sampler(omega, numerical_step, sys)
    for _ in range(50):
        positions = np.random.uniform(-20, 20, (num_particles, num_dimensions))
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        sys.alpha = alpha
        sys.beta = beta
        kine_energy = 0.0
        pos_fw = np.array(positions)
        pos_bw = np.array(positions)
        for i in range(num_particles):
            for j in range(num_dimensions):
                pos_fw[i, j] = pos_fw[i, j] + numerical_step
                pos_bw[i, j] = pos_bw[i, j] - numerical_step
                wf_current = 2*sys.wavefunction(positions)
                wf_forward = sys.wavefunction(pos_fw)
                wf_backwawrd = sys.wavefunction(pos_bw)
                pos_fw[i, j] = pos_fw[i, j] - numerical_step
                pos_bw[i, j] = pos_bw[i, j] + numerical_step
                kine_energy += wf_forward + wf_backwawrd - wf_current
        kine_energy = kine_energy/(numerical_step*numerical_step)

        assert kine_energy == pytest.approx(sam.kinetic_energy(positions),
                                            abs=1e-14)


def test_potential_energy_2d():

    num_particles = 1
    num_dimensions = 2
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
        positions[0, 0] = x
        positions[0, 1] = y
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
    positions = np.zeros(shape=(num_particles, num_dimensions))
    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)
        k = sam.kinetic_energy(positions)/sys.wavefunction(positions)
        p = sam.potential_energy(positions)
        local_energy = -0.5*k + p
        assert local_energy == pytest.approx(sam.local_energy(positions),
                                             abs=1e-14)


def test_local_energy_3d():

    a = 0.0
    num_particles = 1
    num_dimensions = 3
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions[0, 2] = np.random.uniform(-2, 2)
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)
        k = sam.kinetic_energy(positions)/sys.wavefunction(positions)
        p = sam.potential_energy(positions)
        local_energy = -0.5*k + p
        assert local_energy == pytest.approx(sam.local_energy(positions),
                                             abs=1e-14)


def test_local_energy_times_wf_2d():

    a = 0.0
    num_particles = 1
    num_dimensions = 2
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)

        e = sam.local_energy(positions)
        t = sys.derivative_psi_term(positions)*e
        assert t == pytest.approx(sam.local_energy_times_wf(positions),
                                  abs=1e-14)


def test_local_energy_times_wf_3d():

    a = 0.0
    num_particles = 1
    num_dimensions = 3
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions[0, 2] = np.random.uniform(-2, 2)
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)

        e = sam.local_energy(positions)
        t = sys.derivative_psi_term(positions)*e
        assert t == pytest.approx(sam.local_energy_times_wf(positions),
                                  abs=1e-14)


def test_probability_2d():

    a = 0.0
    num_particles = 1
    num_dimensions = 2
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    new_positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        new_positions[0, 0] = np.random.uniform(-2, 2)
        new_positions[0, 1] = np.random.uniform(-2, 2)
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)

        P_old = sys.wavefunction(positions)*sys.wavefunction(positions)
        P_new = sys.wavefunction(new_positions)*sys.wavefunction(new_positions)
        accept_ratio = P_new/P_old
        assert accept_ratio == pytest.approx(sam.probability(positions,
                                             new_positions), abs=1e-14)


def test_probability_3d():

    a = 0.0
    num_particles = 1
    num_dimensions = 3
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    new_positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions[0, 2] = np.random.uniform(-2, 2)
        new_positions[0, 0] = np.random.uniform(-2, 2)
        new_positions[0, 1] = np.random.uniform(-2, 2)
        new_positions[0, 2] = np.random.uniform(-2, 2)
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)

        P_old = sys.wavefunction(positions)*sys.wavefunction(positions)
        P_new = sys.wavefunction(new_positions)*sys.wavefunction(new_positions)
        accept_ratio = P_new/P_old
        assert accept_ratio == pytest.approx(sam.probability(positions,
                                             new_positions), abs=1e-14)


def test_drift_force_2d():

    a = 0.0
    num_particles = 1
    num_dimensions = 2
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    positions_fw = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions_fw[0, 0] = positions[0, 0] + numerical_step
        positions_fw[0, 1] = positions[0, 1] + numerical_step
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)

        wf_current = sys.wavefunction(positions)
        wf_forward = sys.wavefunction(positions_fw)
        deri = (wf_forward - wf_current)/numerical_step
        assert deri == pytest.approx(sam.drift_force(positions), abs=1e-14)


def test_drift_force_3d():

    a = 0.0
    num_particles = 1
    num_dimensions = 3
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    positions_fw = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions[0, 2] = np.random.uniform(-2, 2)
        positions_fw[0, 0] = positions[0, 0] + numerical_step
        positions_fw[0, 1] = positions[0, 1] + numerical_step
        positions_fw[0, 2] = positions[0, 2] + numerical_step
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)

        wf_current = sys.wavefunction(positions)
        wf_forward = sys.wavefunction(positions_fw)
        deri = (wf_forward - wf_current)/numerical_step
        assert deri == pytest.approx(sam.drift_force(positions), abs=1e-14)


def test_greens_function_2d():

    a = 0.0
    num_particles = 1
    num_dimensions = 2
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    new_positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        new_positions[0, 0] = np.random.uniform(-2, 2)
        new_positions[0, 1] = np.random.uniform(-2, 2)
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)

        """
        F_old = sam.drift_force(positions)
        F_new = sam.drift_force(new_positions)
        D = 0.0
        delta_t = 0.001
        G = (0.5*(F_old + F_new) * (0.5 * (positions - new_positions)) +
             D*delta_t*(F_old - F_new))
        G = np.exp(G)
        """
        delta_t = 0.001
        G = 0.0
        assert G == pytest.approx(sam.greens_function(positions, new_positions,
                                                      delta_t), abs=1e-14)


def test_greens_function_3d():

    a = 0.0
    num_particles = 1
    num_dimensions = 3
    numerical_step = 0.001
    positions = np.zeros(shape=(num_particles, num_dimensions))
    new_positions = np.zeros(shape=(num_particles, num_dimensions))

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        positions[0, 2] = np.random.uniform(-2, 2)
        new_positions[0, 0] = np.random.uniform(-2, 2)
        new_positions[0, 1] = np.random.uniform(-2, 2)
        new_positions[0, 2] = np.random.uniform(-2, 2)
        sys = System(num_particles, num_dimensions, alpha, beta, a)
        sam = Sampler(omega, numerical_step, sys)

        """
        F_old = sam.drift_force(positions)
        F_new = sam.drift_force(new_positions)
        D = 0.0
        delta_t = 0.001
        G = (0.5*(F_old + F_new) * (0.5 * (positions - new_positions)) +
             D*delta_t*(F_old - F_new))
        G = np.exp(G)
        """
        delta_t = 0.001
        G = 0.0
        assert G == pytest.approx(sam.greens_function(positions,
                                                      new_positions, delta_t),
                                  abs=1e-14)
