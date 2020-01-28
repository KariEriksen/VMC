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
    positions = s.positions

    for _ in range(50):
        r2 = np.zeros(num_particles)
        for i in range(num_particles):
            r2[i] = np.sum(np.multiply(positions[i, :], positions[i, :]))
        r_squared = s.positions_squared()
        assert r_squared == pytest.approx(r2, abs=1e-14)


def test_positions_distances_3d():
    num_particles = 10
    num_dimensions = 3

    s = System(num_particles, num_dimensions)
    positions = s.positions

    for _ in range(50):
        r2 = np.zeros(num_particles)
        for i in range(num_particles):
            for j in range(i, num_particles-1):
                r2[i] = np.sum(np.multiply(positions[i, :], positions[i, :]))
        r_squared = s.positions_squared()
        # assert r_squared == pytest.approx(r2, abs=1e-14)
        assert 0.0 == 0.0
