# Copyright 2021 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import typing

from pyparsing import *
from collections import defaultdict, namedtuple
from dimod.vartypes import Vartype
from dimod.quadratic import QuadraticModel

MINIMIZE = 1
MAXIMIZE = -1

obj_senses = {"max": MAXIMIZE, "maximum": MAXIMIZE, "maximize": MAXIMIZE,  "min": MINIMIZE, "minimum": MINIMIZE,
              "minimize": MINIMIZE}

GEQ = -1
EQ = 1
LEQ = 1

constraint_senses = {"<": LEQ, "<=": LEQ, "=<": LEQ, "=": EQ,  ">": GEQ, ">=": GEQ, "=>": GEQ}
constraint_symbols = {"<": "<=", "<=": "<=", "=<": "<=", "=": "==",  ">": ">=", ">=": ">=", "=>": ">="}
infinity = 1E30


def make_lp_grammar() -> And:
    """Build the grammar of LP files"""
    # name char ranges for objective, constraint or variable
    all_name_chars = alphanums + "!\"#$%&()/,.;?@_'`{}|~"
    first_char = remove_strings(all_name_chars, nums)
    name = Word(first_char, all_name_chars, max=255)

    # keywords in the LP file
    keywords = ["inf", "infinity",
                "max", "maximum", "maximize",
                "min", "minimum", "minimize",
                "s.t.", "st",
                "bound", "bounds",
                "bin", "binaries", "binary",
                "gen", "general", "generals",
                "end"]

    py_keyword = MatchFirst(map(CaselessKeyword, keywords))
    valid_name = ~py_keyword + name
    valid_name = valid_name.setResultsName("name")

    # second variable for quadratic terms
    second_name = ~py_keyword + name
    second_name = second_name.setResultsName("second_var_name")

    squared_name = ~py_keyword + name
    squared_name = squared_name.setResultsName('squared_name')

    colon = Suppress(oneOf(": ::"))
    plus_minus = oneOf("+ -")
    inf = oneOf("inf infinity", caseless=True)
    number = Word(nums + ".")
    sense = oneOf("< <= =< = > >= =>").setResultsName("sense")

    # section tags
    obj_tag_max = oneOf("max maximum maximize", caseless=True)
    obj_tag_min = oneOf("min minimum minimize", caseless=True)
    obj_tag = (obj_tag_max | obj_tag_min).setResultsName("obj_sense")

    constraints_tag = oneOf(["subj to", "subject to", "s.t.", "st"], caseless=True)

    bounds_tag = oneOf("bound bounds", caseless=True)
    bin_tag = oneOf("bin binaries binary", caseless=True)
    gen_tag = oneOf("gen general generals", caseless=True)
    end_tag = CaselessLiteral("end")

    # coefficient on a variable (includes sign)
    first_var_coef = Optional(plus_minus, "+") + Optional(number, "1")
    first_var_coef.setParseAction(lambda tokens: eval("".join(tokens)))

    coef = plus_minus + Optional(number, "1")
    coef.setParseAction(lambda tokens: eval("".join(tokens)))

    # variable (coefficient and name)
    first_var = Group(first_var_coef.setResultsName("coef") + valid_name)
    var = Group(coef.setResultsName("coef") + valid_name)

    # linear expression
    lin_expr = first_var + ZeroOrMore(var)
    lin_expr = lin_expr.setResultsName("lin_expr")

    # bilinear expression
    quad_vars = Group(first_var_coef.setResultsName("coef") + valid_name + Literal('*').suppress() + second_name)
    # squared expression
    square_vars = Group(
        first_var_coef.setResultsName("coef") + squared_name + Literal('^').suppress() + Literal('2').suppress())

    quadratic_terms = quad_vars | square_vars
    quad_expr = Optional(Literal('+').suppress()) + Literal('[').suppress() + quadratic_terms + \
                ZeroOrMore(quadratic_terms) + Literal(']').suppress()
    quad_expr = quad_expr.setResultsName("quad_expr")

    # for the objective function the standard is having [ quad expression ] / 2
    quad_expr_obj = Optional(Literal('+').suppress()) + Literal('[').suppress() + quadratic_terms + \
                    ZeroOrMore(quadratic_terms) + Literal(']').suppress() + Literal('/').suppress() + \
                    Literal('2').suppress()
    quad_expr_obj = quad_expr_obj.setResultsName("quad_expr")

    # objective
    objective = obj_tag + Optional(valid_name + colon) + Optional(lin_expr) + Optional(quad_expr_obj)
    objective = objective.setResultsName("objective")

    # constraint rhs
    rhs = Optional(plus_minus, "+") + number
    rhs = rhs.setResultsName("rhs")
    rhs.setParseAction(lambda tokens: eval("".join(tokens)))

    # constraints (can be quadratic)
    constraint_word = Group(Optional(valid_name + colon) + Optional(lin_expr) + Optional(quad_expr) + sense + rhs)
    constraints = ZeroOrMore(constraint_word)
    constraints = constraints.setResultsName("constraints")

    # bounds
    signed_inf = (plus_minus + inf).setParseAction(lambda tokens: (tokens[0] == "+") * infinity)
    signed_number = (Optional(plus_minus, "+") + number).setParseAction(lambda tokens: eval("".join(tokens)))
    number_or_inf = (signed_number | signed_inf).setResultsName("numberOrInf")

    # splitting the bounds in left inequality and right inequality
    left_ineq = number_or_inf + sense
    right_ineq = sense + number_or_inf
    bounded_var = Group(Optional(left_ineq).setResultsName("leftbound") + valid_name +
                        Optional(right_ineq).setResultsName("rightbound"))
    free_var = Group(valid_name + Literal("free"))

    bounds_word = free_var | bounded_var
    bounds = bounds_tag + ZeroOrMore(bounds_word).setResultsName("bounds")

    # generals (integer variables)
    generals = gen_tag + ZeroOrMore(valid_name).setResultsName("generals")

    # binaries (binary variables)
    binaries = bin_tag + ZeroOrMore(valid_name).setResultsName("binaries")

    var_info = ZeroOrMore(bounds | generals | binaries)

    # full LP file grammar
    grammar = Optional(objective) + constraints_tag + constraints + var_info + end_tag

    # commenting
    comment_style = Literal("\\") + restOfLine
    grammar.ignore(comment_style)

    return grammar


def remove_strings(string, strings_to_remove):
    """Replace an iterable of strings in removables
       if removables is a string, each character is removed """
    for r in strings_to_remove:
        try:
            string = string.replace(r, "")
        except TypeError:
            raise TypeError("Strings_to_remove contains a non-string element")
    return string


def get_variables_from_parsed_lp(parse_output: ParseResults,
                                 lower_bound_default: typing.Optional[int] = None,
                                 upper_bound_default: typing.Optional[int] = None) -> QuadraticModel:
    """Return a quadratic model containing all the variables included in the CQM

    Args:
        parse_output: the parse results encoding the LP file
        lower_bound_default: the lower bound of the integer variables, in case they are not specified
        upper_bound_default: the upper bound of the integer variables in case they are not specified

    Returns:
        the quadratic model with all variables
    """

    obj = QuadraticModel()

    all_vars = set()
    # default variable type for LP file
    # should be continuous, even though
    # they are not supported in CQMs
    Var = namedtuple('Variable', ['vartype', 'lb', 'ub'])
    variables_info = defaultdict(lambda: Var(vartype="c", lb=lower_bound_default, ub=upper_bound_default))

    # scan the objective
    for oe in parse_output.objective:
        if isinstance(oe, str):
            continue

        else:
            if len(oe) == 2:
                if oe.name != "":
                    all_vars.add(oe.name[0])
                else:
                    all_vars.add(oe.squared_name[0])
            elif len(oe) == 3:
                all_vars.update([oe.name[0], oe.second_var_name[0]])

    # scan the constraints
    for c in parse_output.constraints:

        # scan linear terms
        if c.lin_expr:
            all_vars.update([le.name[0] for le in c.lin_expr])

        # scan quadratic terms of constraints
        if c.quad_expr:
            for qe in c.quad_expr:
                if len(qe) == 3:
                    all_vars.update([qe.name[0] for qe in c.quad_expr])
                    all_vars.update([qe.second_var_name[0] for qe in c.quad_expr])
                elif len(qe) == 2:
                    all_vars.add(qe.squared_name[0])

    # scan the bounds
    for b in parse_output.bounds:

        n = b.name[0]

        # if b.free, default is fine
        if b.leftbound:
            if constraint_senses[b.sense] <= 0:  # NUM >= var
                variables_info[n] = variables_info[n]._replace(ub=b.leftbound[0])
            if constraint_senses[b.sense] >= 0:  # NUM <= var
                variables_info[n] = variables_info[n]._replace(lb=b.leftbound[0])

        if b.rightbound:
            if constraint_senses[b.sense] <= 0:  # var >= NUM
                variables_info[n] = variables_info[n]._replace(lb=b.rightbound[1])

            if constraint_senses[b.sense] >= 0:  # var <= NUM
                variables_info[n] = variables_info[n]._replace(ub=b.rightbound[1])

    # check the binary variables:
    for n in parse_output.binaries:
        variables_info[n] = variables_info[n]._replace(vartype=Vartype.BINARY)

    # check for integer variables
    for n in parse_output.generals:
        variables_info[n] = variables_info[n]._replace(vartype=Vartype.INTEGER)

    for n, var_info in variables_info.items():
        if var_info.vartype is Vartype.BINARY:
            obj.add_variable(Vartype.BINARY, n)
        elif var_info.vartype is Vartype.INTEGER:
            lb = var_info.lb
            ub = var_info.ub

            if lb is not None:
                obj.add_variable(Vartype.INTEGER, n, lower_bound=lb, upper_bound=ub)
            else:
                obj.add_variable(Vartype.INTEGER, n, upper_bound=ub)
        else:
            raise ValueError("Unexpected Vartype: {} for variable: {}".format(var_info.vartype, n))

    return obj
