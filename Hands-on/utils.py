from collections import defaultdict
from itertools import product
from scipy import stats

import numpy as np


def poisson(k, k_max, lam):
    """Truncated Poisson distribution"""

    if (k, k_max, lam) in poisson.cache:
        return poisson.cache[(k, k_max, lam)]

    if k < k_max:
        p = stats.poisson.pmf(k, lam)
    elif k == k_max:
        # basically it's mass of the right tail
        p = 1 - stats.poisson.cdf(k_max - 1, lam)
    else:
        assert False, 'k > k_max'

    poisson.cache[(k, k_max, lam)] = p
    return p


poisson.cache = {}


def get_dynamics(rent_reward=10,
                 expected_req1=3,
                 expected_req2=4,
                 expected_ret1=3,
                 expected_ret2=2,
                 max_parking=20,
                 max_transfer=5):

    T = np.zeros([max_parking + 1, max_parking + 1, max_transfer *
                  2 + 1, max_parking + 1, max_parking + 1, 2])

    def _actions_in_state(n1, n2):
        """Iterator over all actions in state"""
        assert n1 <= max_parking
        assert n2 <= max_parking

        for a in range(min(n1, max_transfer) + 1):
            if (a + n2) > max_parking:
                break
            yield a

        # since a=0 was already yielded, start with 1
        for a in range(1, min(n2, max_transfer) + 1):
            if (a + n1) > max_parking:
                break
            yield -a

    def _simulation(n_cars, expected_req, expected_ret):
        """Iterator over all possible transitions from given number of cars"""
        p_total = 0.0
        for req in range(n_cars + 1):
            p_req = poisson(req, n_cars, expected_req)
            max_ret = max_parking - (n_cars - req)
            for ret in range(max_ret + 1):
                p_ret = poisson(ret, max_ret, expected_ret)
                yield req, p_req, ret, p_ret
                p_total += p_req * p_ret
        assert np.allclose(p_total, 1.), 'Total simulation probability must be 1'

    # For all the states
    for n1, n2 in product(range(max_parking + 1), repeat=2):
        # For all actions
        for a in _actions_in_state(n1, n2):
            # Perform action
            n1_post = n1 - a
            n2_post = n2 + a
            p_total = 0

            # Simulate day in car rentals, iterate over all possibilities
            for req1, p_req1, ret1, p_ret1 in _simulation(n1_post,
                                                          expected_req1,
                                                          expected_ret1):
                for req2, p_req2, ret2, p_ret2 in _simulation(n2_post,
                                                              expected_req2,
                                                              expected_ret2):
                    # Calculate next states
                    n1_next = n1_post - req1 + ret1
                    n2_next = n2_post - req2 + ret2

                    # Checks...
                    assert n1_next >= 0, 'n1_next >= 0'
                    assert n2_next >= 0, 'n2_next >= 0'
                    assert n1_next <= max_parking, 'n1_next <= self.max_parking'
                    assert n2_next <= max_parking, 'n2_next <= self.max_parking'

                    # Next state probability and reward
                    p = p_req1 * p_ret1 * p_req2 * p_ret2
                    r = rent_reward * (req1 + req2)

                    # Update dynamics
                    T[n1, n2, a, n1_next, n2_next] += p, p * r

                    # Book keeping
                    p_total += p
            assert np.allclose(p_total, 1.), 'Total transition probability must be 1'
        print("Set T[({}, {}), ...]".format(n1, n2))

    return T
