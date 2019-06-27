import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from system import System  # noqa: 401


def test_system_wavefunction():
    num_particles = 1
    num_dimensions = 2
    alpha = 1.0
    beta = 1.0
    a = 0.0

    s = System(num_particles, num_dimensions, alpha, beta, a)

    positions = np.random.rand(num_particles, num_dimensions)
    x = 2.92858925782756
    y = 0.00925285752985
    positions[0, 0] = x
    positions[0, 1] = y

    wf = s.single_particle_function(positions)

    assert wf == pytest.approx(np.exp(-x**2 - y**2), abs=1e-14)
