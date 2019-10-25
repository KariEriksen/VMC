import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from Hamiltonian.hamiltonian_testing import Hamiltonian  # noqa: 401
from Hamiltonian.non_interaction import Non_Interaction # noqa: 401
from Hamiltonian.weak_interaction import Weak_Interaction # noqa: 401
from Wavefunction.wavefunction import Wavefunction  # noqa: 401


def test_kinetic_energy_2d():

    num_particles = 1
    num_dimensions = 2
    omega = 1.0
    numerical_step = 0.001
    a = 0.0
    alpha = 0.5
    beta = 1.0
    wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
    ham = Hamiltonian(omega, wave)
    for _ in range(50):
        positions = np.random.uniform(-20, 20, (num_particles, num_dimensions))
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        wave.alpha = alpha
        wave.beta = beta
        kine_energy = 0.0
        pos_fw = np.array(positions)
        pos_bw = np.array(positions)
        for i in range(num_particles):
            for j in range(num_dimensions):
                pos_fw[i, j] = pos_fw[i, j] + numerical_step
                pos_bw[i, j] = pos_bw[i, j] - numerical_step
                wf_current = 2*wave.wavefunction(positions)
                wf_forward = wave.wavefunction(pos_fw)
                wf_backwawrd = wave.wavefunction(pos_bw)
                pos_fw[i, j] = pos_fw[i, j] - numerical_step
                pos_bw[i, j] = pos_bw[i, j] + numerical_step
                kine_energy += wf_forward + wf_backwawrd - wf_current
        kine_energy = kine_energy/(numerical_step*numerical_step)

        assert kine_energy == pytest.approx(ham.laplacian_numerical(positions),
                                            abs=1e-8)


def test_kinetic_energy_2d_2p():

    num_particles = 2
    num_dimensions = 2
    omega = 1.0
    numerical_step = 0.001
    a = 0.0
    alpha = 0.5
    beta = 1.0
    wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
    ham = Hamiltonian(omega, wave)
    for _ in range(50):
        positions = np.random.uniform(-20, 20, (num_particles, num_dimensions))
        alpha = np.random.uniform(1e-2, 10)
        beta = np.random.uniform(1e-3, 10)
        wave.alpha = alpha
        wave.beta = beta
        kine_energy = 0.0
        pos_fw = np.array(positions)
        pos_bw = np.array(positions)
        for i in range(num_particles):
            for j in range(num_dimensions):
                pos_fw[i, j] = pos_fw[i, j] + numerical_step
                pos_bw[i, j] = pos_bw[i, j] - numerical_step
                wf_current = 2*wave.wavefunction(positions)
                wf_forward = wave.wavefunction(pos_fw)
                wf_backwawrd = wave.wavefunction(pos_bw)
                pos_fw[i, j] = pos_fw[i, j] - numerical_step
                pos_bw[i, j] = pos_bw[i, j] + numerical_step
                kine_energy += wf_forward + wf_backwawrd - wf_current
        kine_energy = kine_energy/(numerical_step*numerical_step)

        assert kine_energy == pytest.approx(ham.laplacian_numerical(positions),
                                            abs=1e-8)


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
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        ham = Hamiltonian(omega, wave)
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

        wf_current = 2*num_dimensions*wave.wavefunction(positions)
        wf_forward = (wave.wavefunction(pos_xp) + wave.wavefunction(pos_yp) +
                      wave.wavefunction(pos_zp))
        wf_backwawrd = (wave.wavefunction(pos_xn) + wave.wavefunction(pos_yn) +
                        wave.wavefunction(pos_zn))
        kine_energy = wf_forward + wf_backwawrd - wf_current
        kine_energy = kine_energy/(numerical_step*numerical_step)

        assert kine_energy == pytest.approx(ham.laplacian_numerical(positions),
                                            abs=1e-8)


def test_kinetic_energy_3d_2p():

    num_particles = 2
    num_dimensions = 3
    omega = 1.0
    numerical_step = 0.001
    a = 0.0
    alpha = 0.5
    beta = 1.0
    wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
    ham = Hamiltonian(omega, wave)
    for _ in range(50):
        positions = np.random.uniform(-20, 20, (num_particles, num_dimensions))
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        wave.alpha = alpha
        wave.beta = beta
        kine_energy = 0.0
        pos_fw = np.array(positions)
        pos_bw = np.array(positions)
        for i in range(num_particles):
            for j in range(num_dimensions):
                pos_fw[i, j] = pos_fw[i, j] + numerical_step
                pos_bw[i, j] = pos_bw[i, j] - numerical_step
                wf_current = 2*wave.wavefunction(positions)
                wf_forward = wave.wavefunction(pos_fw)
                wf_backwawrd = wave.wavefunction(pos_bw)
                pos_fw[i, j] = pos_fw[i, j] - numerical_step
                pos_bw[i, j] = pos_bw[i, j] + numerical_step
                kine_energy += wf_forward + wf_backwawrd - wf_current
        kine_energy = kine_energy/(numerical_step*numerical_step)

        assert kine_energy == pytest.approx(ham.laplacian_numerical(positions),
                                            abs=1e-8)


def test_kinetic_analytic_2d():
    # test for kinetic energy in the
    # non-non_interacting case using analytic expression

    num_particles = 1
    num_dimensions = 2
    positions = np.zeros(shape=(num_particles, num_dimensions))
    a = 0.0
    beta = 1.0
    omega = 1.0
    for _ in range(50):
        energy = 0.0
        alpha = np.random.uniform(1e-3, 10)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        ham = Non_Interaction(omega, wave, True)
        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        positions[0, 0] = x
        positions[0, 1] = y
        energy = 4*alpha*alpha*(x*x + y*y) - 4*alpha
        assert energy == pytest.approx(ham.laplacian_analytical
                                       (positions), abs=1e-14)


def test_kinetic_analytic_2d_2p():
    # test for kinetic energy in the
    # non-non_interacting case using analytic expression

    num_particles = 2
    num_dimensions = 2
    positions = np.zeros(shape=(num_particles, num_dimensions))
    a = 0.0
    beta = 1.0
    omega = 1.0
    for _ in range(50):
        energy = 0.0
        alpha = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        ham = Non_Interaction(omega, wave, True)
        x1 = np.random.uniform(-20, 20)
        y1 = np.random.uniform(-20, 20)
        x2 = np.random.uniform(-20, 20)
        y2 = np.random.uniform(-20, 20)
        positions[0, 0] = x1
        positions[0, 1] = y1
        positions[1, 0] = x2
        positions[1, 1] = y2
        for i in range(num_particles):
            x = positions[i, 0]
            y = positions[i, 1]
            energy += 4*alpha*alpha*(x*x + y*y) - 4*alpha
        assert energy == pytest.approx(ham.laplacian_analytical
                                       (positions), abs=1e-10)


def test_kinetic_analytic_3d():

    num_particles = 1
    num_dimensions = 3
    positions = np.zeros(shape=(num_particles, num_dimensions))
    a = 0.0
    beta = 1.0
    omega = 1.0
    for _ in range(50):
        energy = 0.0
        alpha = np.random.uniform(1e-3, 10)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        ham = Non_Interaction(omega, wave, True)
        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-20, 20)
        positions[0, 0] = x
        positions[0, 1] = y
        positions[0, 2] = z
        energy = 4*alpha*alpha*(x*x + y*y + z*z) - 6*alpha
        assert energy == pytest.approx(ham.laplacian_analytical
                                       (positions), abs=1e-14)


def test_kinetic_analytic_3d_2p():

    num_particles = 2
    num_dimensions = 3
    positions = np.zeros(shape=(num_particles, num_dimensions))
    a = 0.0
    beta = 1.0
    omega = 1.0
    for _ in range(50):
        energy = 0.0
        alpha = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        ham = Non_Interaction(omega, wave, True)
        x1 = np.random.uniform(-20, 20)
        y1 = np.random.uniform(-20, 20)
        z1 = np.random.uniform(-20, 20)
        x2 = np.random.uniform(-20, 20)
        y2 = np.random.uniform(-20, 20)
        z2 = np.random.uniform(-20, 20)
        positions[0, 0] = x1
        positions[0, 1] = y1
        positions[1, 1] = z1
        positions[1, 0] = x2
        positions[1, 1] = y2
        positions[1, 1] = z2
        for i in range(num_particles):
            x = positions[i, 0]
            y = positions[i, 1]
            z = positions[i, 2]
            energy += 4*alpha*alpha*(x*x + y*y + z*z) - 6*alpha
        assert energy == pytest.approx(ham.laplacian_analytical
                                       (positions), abs=1e-10)


def test_potential_energy_2d():

    num_particles = 1
    num_dimensions = 2
    a = 0.0
    positions = np.zeros(shape=(num_particles, num_dimensions))
    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        ham = Hamiltonian(omega, wave)

        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        positions[0, 0] = x
        positions[0, 1] = y
        sum = 0.0
        for i in range(num_dimensions):
            sum += (positions[0, i]*positions[0, i])

        pot_energy = 0.5*omega*omega*(sum)
        assert pot_energy == pytest.approx(ham.trap_potential_energy
                                           (positions), abs=1e-14)


def test_potential_energy_2d_2p():

    num_particles = 2
    num_dimensions = 2
    a = 0.0
    positions = np.zeros(shape=(num_particles, num_dimensions))
    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        ham = Hamiltonian(omega, wave)

        x1 = np.random.uniform(-20, 20)
        y1 = np.random.uniform(-20, 20)
        x2 = np.random.uniform(-20, 20)
        y2 = np.random.uniform(-20, 20)
        positions[0, 0] = x1
        positions[0, 1] = y1
        positions[1, 0] = x2
        positions[1, 1] = y2

        sum = x1*x1 + x2*x2 + y1*y1 + y2*y2

        pot_energy = 0.5*omega*omega*(sum)
        assert pot_energy == pytest.approx(ham.trap_potential_energy
                                           (positions), abs=1e-10)


def test_potential_energy_3d():

    num_particles = 1
    num_dimensions = 3
    a = 0.0
    positions = np.zeros(shape=(num_particles, num_dimensions))
    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        ham = Hamiltonian(omega, wave)

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
        assert pot_energy == pytest.approx(ham.trap_potential_energy
                                           (positions), abs=1e-14)


def test_potential_energy_3d_2p():

        num_particles = 2
        num_dimensions = 3
        a = 0.0
        positions = np.zeros(shape=(num_particles, num_dimensions))
        for _ in range(50):
            alpha = np.random.uniform(1e-3, 10)
            omega = np.random.uniform(1e-3, 10)
            wave = Wavefunction(num_particles, num_dimensions, alpha, 1.0, a)
            ham = Hamiltonian(omega, wave)

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

            sum = x1*x1 + x2*x2 + y1*y1 + y2*y2 + z1*z1 + z2*z2

            pot_energy = 0.5*omega*omega*(sum)
            assert pot_energy == pytest.approx(ham.trap_potential_energy
                                               (positions), abs=1e-10)


def test_local_energy_2d():

    a = 0.0
    num_particles = 1
    num_dimensions = 2
    positions = np.zeros(shape=(num_particles, num_dimensions))
    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        beta = np.random.uniform(1e-3, 10)
        omega = np.random.uniform(1e-3, 10)
        positions[0, 0] = np.random.uniform(-2, 2)
        positions[0, 1] = np.random.uniform(-2, 2)
        wave = Wavefunction(num_particles, num_dimensions, alpha, beta, a)
        ham = Non_Interaction(omega, wave, True)
        k = ham.laplacian_analytical(positions)/wave.wavefunction(positions)
        p = ham.trap_potential_energy(positions)
        local_energy = -0.5*k + p
        test_loc_ener = ham.local_energy(positions)
        assert local_energy == pytest.approx(test_loc_ener, abs=1e-14)


def test_local_energy_3d():

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
        ham = Non_Interaction(omega, wave, True)
        k = ham.laplacian_analytical(positions)/wave.wavefunction(positions)
        p = ham.trap_potential_energy(positions)
        local_energy = -0.5*k + p
        test_loc_ener = ham.local_energy(positions)
        assert local_energy == pytest.approx(test_loc_ener, abs=1e-14)
