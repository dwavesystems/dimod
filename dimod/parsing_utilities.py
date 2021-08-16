from pyparsing import *

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


def parse_lp_file(input_string: str) -> ParseResults:
    """"""
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
    grammar = objective + constraints_tag + constraints + var_info + end_tag

    # commenting
    comment_style = Literal("\\") + restOfLine
    grammar.ignore(comment_style)

    # parse input string
    parse_output = grammar.parseString(input_string)

    return parse_output


def remove_strings(string, strings_to_remove):
    """Replaces an iterable of strings in removables
        if removables is a string, each character is removed """
    for r in strings_to_remove:
        try:
            string = string.replace(r, "")
        except TypeError:
            raise TypeError("Strings_to_remove contains a non-string element")
    return string