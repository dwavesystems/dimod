def ising_energy(h, J, solution):
    """TODO"""

    en = 0

    for v in h:
        en += h[v] * solution[v]

    for v0, v1 in J:
        en += J[(v0, v1)] * solution[v0] * solution[v1]

    return en
