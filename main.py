from vmc import non_interaction_case # noqa: 401
from vmc import weak_interaction_case # noqa: 401
from vmc import strong_interaction_case # noqa: 401
from vmc import elliptic_weak_interaction_case # noqa: 401
from vmc import brute_force # noqa: 401
from vmc import one_body_density # noqa: 401
from vmc import run_blocking # noqa: 401


"""case(monte_carlo_cycles, number of particles,
        number of dimensions, interaction parameter)"""

# non_interaction_case(int(10e4), 1, 3, 0.4)
# weak_interaction_case(100000, 2, 3, 0.4)
strong_interaction_case(100000, 3, 3, 0.4)
# elliptic_weak_interaction_case(10000, 2, 3, None)
# brute_force(100000, 1, 3, None)
# one_body_density(100000, 3, 3, None)
# run_blocking(pow(2, 24), 2, 3, 0.495)
