from collections import OrderedDict

from sympy import collect, collect_const

from devito.ir.dfg import temporaries_graph
from devito.symbolics import Eq, count, estimate_cost, q_op, q_leaf, xreplace_constrained
from devito.types import Indexed, Array
from devito.tools import flatten

__all__ = ['promote_scalar_expressions', 'collect_nested',
           'common_subexprs_elimination', 'compact_temporaries']


def promote_scalar_expressions(exprs, shape, indices, onstack):
    """
    Transform a collection of scalar expressions into tensor expressions.
    """
    processed = []

    # Fist promote the LHS
    graph = temporaries_graph(exprs)
    mapper = {}
    for k, v in graph.items():
        if v.is_scalar:
            # Create a new function symbol
            data = Array(name=k.name, shape=shape,
                         dimensions=indices, onstack=onstack)
            indexed = Indexed(data.indexed, *indices)
            mapper[k] = indexed
            processed.append(Eq(indexed, v.rhs))
        else:
            processed.append(Eq(k, v.rhs))

    # Propagate the transformed LHS through the expressions
    processed = [Eq(n.lhs, n.rhs.xreplace(mapper)) for n in processed]

    return processed


def collect_nested(expr, aggressive=False):
    """
    Collect terms appearing in expr, checking all levels of the expression tree.

    :param expr: the expression to be factorized.
    """

    def run(expr):
        # Return semantic (rebuilt expression, factorization candidates)

        if expr.is_Number or expr.is_Symbol:
            return expr, [expr]
        elif expr.is_Indexed or expr.is_Atom:
            return expr, []
        elif expr.is_Add:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])

            w_numbers = [i for i in rebuilt if any(j.is_Number for j in i.args)]
            wo_numbers = [i for i in rebuilt if i not in w_numbers]

            w_numbers = collect_const(expr.func(*w_numbers))
            wo_numbers = expr.func(*wo_numbers)

            if aggressive is True and wo_numbers:
                for i in flatten(candidates):
                    wo_numbers = collect(wo_numbers, i)

            rebuilt = expr.func(w_numbers, wo_numbers)
            return rebuilt, []
        elif expr.is_Mul:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])
            rebuilt = collect_const(expr.func(*rebuilt))
            return rebuilt, flatten(candidates)
        else:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])
            return expr.func(*rebuilt), flatten(candidates)

    return run(expr)[0]


def common_subexprs_elimination(exprs, make, mode='default'):
    """
    Perform common subexpressions elimination.

    Note: the output is not guranteed to be topologically sorted.

    :param exprs: The target SymPy expression, or a collection of SymPy expressions.
    :param make: A function to construct symbols used for replacement.
                 The function takes as input an integer ID; ID is computed internally
                 and used as a unique identifier for the constructed symbols.
    """

    # Note: not defaulting to SymPy's CSE() function for three reasons:
    # - it also captures array index access functions (eg, i+1 in A[i+1] and B[i+1]);
    # - it sometimes "captures too much", losing factorization opportunities;
    # - very slow
    # TODO: a second "sympy" mode will be provided, relying on SymPy's CSE() but
    # also ensuring some sort of post-processing
    assert mode == 'default'  # Only supported mode ATM

    processed = list(exprs)
    mapped = []
    while True:
        # Detect redundancies
        counted = count(mapped + processed, q_op).items()
        targets = OrderedDict([(k, estimate_cost(k)) for k, v in counted if v > 1])
        if not targets:
            break

        # Create temporaries
        hit = max(targets.values())
        picked = [k for k, v in targets.items() if v == hit]
        mapper = OrderedDict([(e, make(len(mapped) + i)) for i, e in enumerate(picked)])

        # Apply repleacements
        processed = [e.xreplace(mapper) for e in processed]
        mapped = [e.xreplace(mapper) for e in mapped]
        mapped = [Eq(v, k) for k, v in reversed(list(mapper.items()))] + mapped

        # Prepare for the next round
        for k in picked:
            targets.pop(k)
    processed = mapped + processed

    # Simply renumber the temporaries in ascending order
    mapper = {i.lhs: j.lhs for i, j in zip(mapped, reversed(mapped))}
    processed = [e.xreplace(mapper) for e in processed]

    return processed


def compact_temporaries(temporaries, leaves):
    """
    Drop temporaries consisting of single symbols.
    """
    exprs = temporaries + leaves
    targets = {i.lhs for i in leaves}

    g = temporaries_graph(exprs)

    mapper = {k: v.rhs for k, v in g.items()
              if v.is_scalar and
              (q_leaf(v.rhs) or v.rhs.is_Function) and
              not v.readby.issubset(targets)}

    processed = []
    for k, v in g.items():
        if k not in mapper:
            # The temporary /v/ is retained, and substitutions may be applied
            handle, _ = xreplace_constrained(v, mapper, repeat=True)
            assert len(handle) == 1
            processed.extend(handle)

    return processed
