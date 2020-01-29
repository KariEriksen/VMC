import pytest
import numpy as np
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from system import System  # noqa: 401


def test_positions_squared_3d():
    num_particles = 10
    num_dimensions = 3

    s = System(num_particles, num_dimensions)
    positions = np.random.rand(num_particles, num_dimensions)

    for _ in range(50):
        r2 = np.zeros(num_particles)
        for i in range(num_particles):
            r2[i] = np.sum(np.multiply(positions[i, :], positions[i, :]))
        r_squared = s.positions_squared(positions)
        assert r_squared == pytest.approx(r2, abs=1e-14)


def test_positions_distances_3d():
    num_particles = 3
    num_dimensions = 3

    s = System(num_particles, num_dimensions)
    positions = np.random.rand(num_particles, num_dimensions)

    for _ in range(50):
        d = np.zeros((num_particles, num_particles))

        r12 = np.subtract(positions[0, :], positions[1, :])
        r13 = np.subtract(positions[0, :], positions[2, :])
        r23 = np.subtract(positions[1, :], positions[2, :])
        r12 = r12*r12
        r13 = r13*r13
        r23 = r23*r23
        d[0, 1] = math.sqrt(np.sum(r12))
        d[0, 2] = math.sqrt(np.sum(r13))
        d[1, 2] = math.sqrt(np.sum(r23))
        # r2[1, 0] = r2[0, 1]
        # r2[2, 0] = r2[0, 2]
        # r2[2, 1] = r2[1, 2]
        r_distance = s.positions_distances(positions)
        assert r_distance == pytest.approx(d, abs=1e-14)


def test_distances_update_3d():
    num_particles = 3
    num_dimensions = 3

    s = System(num_particles, num_dimensions)
    positions = np.random.rand(num_particles, num_dimensions)

    for _ in range(50):
        d = np.zeros((num_particles, num_particles))

        r12 = np.subtract(positions[0, :], positions[1, :])
        r13 = np.subtract(positions[0, :], positions[2, :])
        r23 = np.subtract(positions[1, :], positions[2, :])
        r12 = r12*r12
        r13 = r13*r13
        r23 = r23*r23
        d[0, 1] = math.sqrt(np.sum(r12))
        d[0, 2] = math.sqrt(np.sum(r13))
        d[1, 2] = math.sqrt(np.sum(r23))
        d[1, 0] = d[0, 1]
        # d[2, 0] = d[0, 2]
        d[2, 1] = d[1, 2]
        s.positions_distances(positions)
        r_distance = s.distances_update(positions, 1)
        assert r_distance == pytest.approx(d, abs=1e-14)
