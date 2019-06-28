import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
from optimizer import Optimizer  # noqa: 401


def test_gradient_descent():
    learning_rate = 0.01
    alpha = np.random.uniform(1e-3, 10)
    derivative_energy = np.random.uniform(1e-5, 100)

    o = Optimizer(learning_rate)
    new_alpha = o.gradient_descent(alpha, derivative_energy)

    assert new_alpha == pytest.approx(alpha - learning_rate*derivative_energy,
                                      abs=1e-14)

    for _ in range(50):
        alpha = np.random.uniform(1e-3, 10)
        derivative_energy = np.random.uniform(1e-5, 100)

        new_alpha = o.gradient_descent(alpha, derivative_energy)
        assert new_alpha == pytest.approx(alpha - learning_rate *
                                          derivative_energy, abs=1e-14)
