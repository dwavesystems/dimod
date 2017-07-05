import itertools
import random
import math


def simulated_annealing_simple(h, J, steps=5000):

    nodes = list(h.keys())

    energies = {}

    def energy_difference(s0, s1):
        diff = 0

        for v in h:
            if s0[v] > s1[v]:
                diff -= h[v]
            elif s0[v] < s1[v]:
                diff += h[v]

        for v0, v1 in J:
            if s0[v0] != s1[v0] or s0[v1] != v1[v1]:
                diff += J[(v0, v1)] * (s0[v0] * s0[v1] - s1[v0] * s1[v1])

        return diff

    s = {v: bias > 0 and -1 or 1 for v, bias in h.items()}
    energy = ising_energy(s)

    for k in range(1, steps):
        T = float(k) / steps

        sp = neighbor(s)
        energy_p = ising_energy(sp)

        if accept(diff, 0, T):
            s = sp
            energy = energy_p

    return s


def accept(en, enp, T):
    if enp < en:
        return True

    p = math.exp(-1 * float(enp - en) / T)
    return random.random() < p


def neighbor(s, hamming_distance=1):
    flips = random.choice(list(itertools.combinations(s, hamming_distance)))
    sp = s.copy()

    for v in flips:
        sp[v] *= -1

    return sp


if __name__ == '__main__':
    nV = 100
    h = {v: random.uniform(-2, 2) for v in range(nV)}
    J = {(v0, v1): random.uniform(-1, 1) for v0, v1 in itertools.combinations(range(nV), 2)
         if random.choice((0, 1))}

    s = simulated_annealing_simple(h, J)

    print s
