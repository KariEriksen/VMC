from vmc import non_interaction_case # noqa: 401
from vmc import weak_interaction_case # noqa: 401
from vmc import strong_interaction_case # noqa: 401
from vmc import elliptic_weak_interaction_case # noqa: 401
from vmc import brute_force # noqa: 401
from vmc import one_body_density # noqa: 401
from vmc import run_blocking # noqa: 401
import numpy as np

# np.random.seed(10)

"""case(monte_carlo_cycles, number of particles,
        number of dimensions, interaction parameter)"""

# non_interaction_case(int(10e6), 1, 3, 0.1)
# weak_interaction_case(1000, 2, 3, 0.49)
# strong_interaction_case(10, 2, 3, 2.8)
# elliptic_weak_interaction_case(10000, 2, 3, None)
# brute_force(100000, 1, 3, None)
one_body_density(10000, 3, 3, None)
# run_blocking(pow(2, 24), 2, 3, 0.4997)
