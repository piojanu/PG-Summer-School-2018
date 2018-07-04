from collections import defaultdict
from itertools import product
from os.path import isfile
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

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


class MDP(object):
    def __init__(self,
                 rent_reward=10,
                 transfer_cost=2,
                 expected_req1=3,
                 expected_req2=4,
                 expected_ret1=3,
                 expected_ret2=2,
                 max_parking=20,
                 max_transfer=5):
        self.rent_reward = rent_reward
        self.transfer_cost = transfer_cost
        self.expected_req1 = expected_req1
        self.expected_req2 = expected_req2
        self.expected_ret1 = expected_ret1
        self.expected_ret2 = expected_ret2
        self.max_parking = max_parking
        self.max_transfer = max_transfer


    def actions_in_state(self, n1, n2):
        """Iterator over all actions in state"""
        assert n1 <= self.max_parking
        assert n2 <= self.max_parking

        for a in range(min(n1, self.max_transfer) + 1):
            if (a + n2) > self.max_parking:
                break
            yield a

        # since a=0 was already yielded, start with 1
        for a in range(1, min(n2, self.max_transfer) + 1):
            if (a + n1) > self.max_parking:
                break
            yield -a



    def get_dynamics(self, path=None):

        if path is not None and isfile(path):
            return np.load(path)

        T = np.zeros([self.max_parking + 1, self.max_parking + 1, self.max_transfer *
                      2 + 1, self.max_parking + 1, self.max_parking + 1, 2])

        def _simulation(n_cars, expected_req, expected_ret):
            """Iterator over all possible transitions from given number of cars"""
            p_total = 0.0
            for req in range(n_cars + 1):
                p_req = poisson(req, n_cars, expected_req)
                max_ret = self.max_parking - (n_cars - req)
                for ret in range(max_ret + 1):
                    p_ret = poisson(ret, max_ret, expected_ret)
                    yield req, p_req, ret, p_ret
                    p_total += p_req * p_ret
            assert np.allclose(p_total, 1.), 'Total simulation probability must be 1'

        # For all the states
        for n1, n2 in product(range(self.max_parking + 1), repeat=2):
            # For all actions
            for a in self.actions_in_state(n1, n2):
                # Perform action
                cost = self.transfer_cost * abs(a)
                n1_post = n1 - a
                n2_post = n2 + a
                p_total = 0

                # Simulate day in car rentals, iterate over all possibilities
                for req1, p_req1, ret1, p_ret1 in _simulation(n1_post,
                                                              self.expected_req1,
                                                              self.expected_ret1):
                    for req2, p_req2, ret2, p_ret2 in _simulation(n2_post,
                                                                  self.expected_req2,
                                                                  self.expected_ret2):
                        # Calculate next states
                        n1_next = n1_post - req1 + ret1
                        n2_next = n2_post - req2 + ret2

                        # Checks...
                        assert n1_next >= 0, 'n1_next >= 0'
                        assert n2_next >= 0, 'n2_next >= 0'
                        assert n1_next <= self.max_parking, 'n1_next <= self.max_parking'
                        assert n2_next <= self.max_parking, 'n2_next <= self.max_parking'

                        # Next state probability and reward
                        p = p_req1 * p_ret1 * p_req2 * p_ret2
                        r = self.rent_reward * (req1 + req2) - cost

                        # Update dynamics
                        T[n1, n2, a, n1_next, n2_next][0] += p
                        T[n1, n2, a, n1_next, n2_next][1] += p * r

                        # Book keeping
                        p_total += p

                # Normalize probabilities
                for n1_next, n2_next in product(range(self.max_parking + 1), repeat=2):
                    T[n1, n2, a, n1_next, n2_next][1] /= T[n1, n2, a, n1_next, n2_next][0]

                assert np.allclose(p_total, 1.), 'Total transition probability must be 1'

        # Save generated dynamics
        if path is not None:
            np.save(path, T)

        return T

    def states_iter(self):
        for n1, n2 in product(range(self.max_parking + 1), repeat=2):
            yield n1, n2


def visualize(pi=None, V=None):
    fig = plt.figure(figsize=plt.figaspect(0.4))

    if V is not None:
        ax = fig.add_subplot(1, 2, 1)
        sns.heatmap(V, vmin=350, vmax=650, ax=ax)
        ax.set_ylabel("1st location")
        ax.set_xlabel("2nd location")
        ax.set_title("State values")

    if pi is not None:
        ax = fig.add_subplot(1, 2, 2)
        sns.heatmap(pi, annot=True, ax=ax, cbar=False)
        ax.set_ylabel("1st location")
        ax.set_xlabel("2nd location")
        ax.set_title("Current policy")
