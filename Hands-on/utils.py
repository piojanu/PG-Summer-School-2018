import numpy as np

lake_annot = [
    np.array([
        ['S', 'F', 'F', 'F'],
        ['F', 'H', 'F', 'H'],
        ['F', 'F', 'F', 'H'],
        ['H', 'F', 'F', 'G']]),
    np.array([
        ['S', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
        ['F', 'F', 'F', 'F', 'F', 'H', 'F', 'F'],
        ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
        ['F', 'H', 'H', 'F', 'F', 'F', 'H', 'F'],
        ['F', 'H', 'F', 'F', 'H', 'F', 'H', 'F'],
        ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'G']])
]


def action2str(a):
    if a == 0:
        return 'L'
    elif a == 1:
        return 'D'
    elif a == 2:
        return 'R'
    elif a == 3:
        return 'U'
    else:
        return '-'


def vi_frozen_lake(env):
    # MDP transition/dynamics
    P = env.env.P
    n_states, n_actions = env.observation_space.n, env.action_space.n

    V = np.zeros(n_states)

    while True:
        delta = 0
        # Start from the end to speed value propagation
        for s in reversed(range(n_states)):
            oldV = V[s]
            V[s] = np.max([sum([p*(r + V[s_]) for p, s_, r, _ in P[s][a]])
                           for a in range(n_actions)])
            delta = max(delta, abs(oldV - V[s]))

        if delta < 1e-30:
            break

    pi_str = []
    for s in range(n_states):
        pi = np.argmax([sum([p*(r + V[s_]) for p, s_, r, _ in P[s][a]]) for a in range(n_actions)])
        pi_str.append(action2str(pi if len(P[s][0]) > 1 else 5))

    # Overide value of goal state to make graph nicer
    V[-1] = 1

    return V, pi_str
