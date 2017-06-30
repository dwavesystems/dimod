class DMResponse(object):
    def __init__(self, solutions):
        self.solutions = solutions

    def __iter__(self):
        for solution in self.solutions:
            yield solution

    def __getitem__(self, index_key):
        return self.solutions[index_key]

    def relabel_variables(self, mapping):

        relabelled_solutions = [{mapping[var]: solution[var] for var in solution}
                                for solution in self.solutions]

        return self.__class__(relabelled_solutions)


class BooleanResponse(DMResponse):
    def as_spins(self):
        spin_solutions = [{var: 2 * solution[var] - 1 for var in solution}
                          for solution in self.solutions]

        return SpinResponse(spin_solutions)


class SpinResponse(DMResponse):
    def as_bool(self):
        bool_solutions = [{var: (solution[var] + 1) / 2 for var in solution}
                          for solution in self.solutions]

        return BooleanResponse(bool_solutions)
