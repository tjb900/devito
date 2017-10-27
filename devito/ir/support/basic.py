from sympy import Basic, Eq

from devito.symbolics import retrieve_indexed
from devito.tools import as_tuple, is_integer, filter_sorted
from devito.types import Indexed


class Vector(tuple):

    """
    A representation of a vector in Z^n.

    The elements of a Vector can be integers or any SymPy expression.

    Notes on Vector comparison
    ==========================

    # Vector-scalar comparison
    --------------------------
    If a comparison between a vector and a non-vector is attempted, then the
    non-vector is promoted to a vector; if this is not possible, an exception
    is raised. This is handy because it turns a vector-scalar comparison into
    a vector-vector comparison with the scalar broadcasted to all vector entries.
    For example: ::

        (3, 4, 5) > 4 => (3, 4, 5) > (4, 4, 4) => False

    # Comparing Vector entries when these are SymPy expression
    ----------------------------------------------------------
    When we compare two entries that are both generic SymPy expressions, it is
    generally not possible to determine the truth value of the relation. For
    example, the truth value of `3*i < 4*j` cannot be determined. In some cases,
    however, the comparison is feasible; for example, `i + 4 < i` should always
    return false. A sufficient condition for two Vectors to be comparable is that
    their pair-wise indices are affine functions of the same variables, with
    coefficient 1.
    """

    def __new__(cls, *items):
        if not all(is_integer(i) or isinstance(i, Basic) for i in items):
            raise TypeError("Illegal Vector element type")
        return super(Vector, cls).__new__(cls, items)

    def _asvector(func):
        def wrapper(self, other):
            if not isinstance(other, Vector):
                try:
                    other = Vector(*other)
                except TypeError:
                    # Not iterable
                    other = Vector(*(as_tuple(other)*len(self)))
            if len(self) != len(other):
                raise TypeError("Cannot operate with Vectors of different rank")
            return func(self, other)
        return wrapper

    @_asvector
    def __add__(self, other):
        return Vector(*[i + j for i, j in zip(self, other)])

    @_asvector
    def __radd__(self, other):
        return self + other

    @_asvector
    def __sub__(self, other):
        return Vector(*[i - j for i, j in zip(self, other)])

    @_asvector
    def __rsub__(self, other):
        return self - other

    @_asvector
    def __eq__(self, other):
        return super(Vector, self).__eq__(other)

    @_asvector
    def __ne__(self, other):
        return super(Vector, self).__ne__(other)

    @_asvector
    def __lt__(self, other):
        try:
            diff = [int(i) for i in self.order(other)]
        except TypeError:
            raise TypeError("Cannot compare due to non-comparable index functions")
        return diff < [0]*self.rank

    @_asvector
    def __gt__(self, other):
        return other.__lt__(self)

    @_asvector
    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    @_asvector
    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __getitem__(self, key):
        ret = super(Vector, self).__getitem__(key)
        return Vector(*ret) if isinstance(key, slice) else ret

    def __repr__(self):
        maxlen = max(3, max([len(str(i)) for i in self]))
        return '\n'.join([('|{:^%d}|' % maxlen).format(str(i)) for i in self])

    @property
    def rank(self):
        return len(self)

    @property
    def sum(self):
        return sum(self)

    def distance(self, other):
        """Compute vector distance from ``self`` to ``other``."""
        return self - other

    def order(self, other):
        """
        A reflexive, transitive, and anti-symmetric relation for total ordering.

        Return a tuple of length equal to the Vector ``rank``. The i-th tuple
        entry, of type int, indicates whether the i-th component of ``self``
        precedes (< 0), equals (== 0), or succeeds (> 0) the i-th component of
        ``other``.
        """
        return self.distance(other)


class IterationInstance(Vector):

    """A representation of the iteration space point accessed by a
    :class:`Indexed` object."""

    def __new__(cls, indexed):
        assert isinstance(indexed, Indexed)
        obj = super(IterationInstance, cls).__new__(cls, *indexed.indices)
        obj.findices = tuple(indexed.base.function.indices)
        return obj

    def __eq__(self, other):
        if isinstance(other, IterationInstance) and self.findices != other.findices:
            raise TypeError("Cannot compare due to mismatching `findices`")
        return super(IterationInstance, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, IterationInstance) and self.findices != other.findices:
            raise TypeError("Cannot compare due to mismatching `findices`")
        return super(IterationInstance, self).__lt__(other)

    def __gt__(self, other):
        return other.__lt__(self)

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def distance(self, sink, dim=None):
        """Compute vector distance from ``self`` to ``sink``. If ``dim`` is
        supplied, compute the vector distance up to and including ``dim``."""
        if not isinstance(sink, IterationInstance):
            raise TypeError("Cannot compute distance from obj of type %s", type(sink))
        if self.findices != sink.findices:
            raise TypeError("Cannot compute distance due to mismatching `findices`")
        if dim is not None:
            try:
                limit = self.findices.index(dim) + 1
            except ValueError:
                raise TypeError("Cannot compute distance as `dim` is not in `findices`")
        else:
            limit = self.rank
        return super(IterationInstance, self).distance(sink)[:limit]


class Access(IterationInstance):

    """
    A representation of the access performed by a :class:`Indexed` object.

    Notes on Access comparison
    ==========================

    The comparison operators ``==, !=, <, <=, >, >=`` should be regarded as
    operators for lexicographic ordering of :class:`Access` objects, based
    on the values of the access functions (and the access functions only).

    For example, if two Access objects A and B employ the same index functions,
    the operation A == B will return True regardless of whether A and B are
    reads or writes or mixed.
    """

    def __new__(cls, indexed, mode):
        assert isinstance(indexed, Indexed)
        assert mode in ['R', 'W']
        obj = super(Access, cls).__new__(cls, indexed)
        obj.function = indexed.base.function
        obj.mode = mode
        return obj

    def __eq__(self, other):
        return super(Access, self).__eq__(other) and\
            isinstance(other, Access) and\
            self.function == other.function

    @property
    def is_read(self):
        return self.mode == 'R'

    @property
    def is_write(self):
        return self.mode == 'W'

    def __repr__(self):
        mode = '\033[1;37;31mW\033[0m' if self.is_write else '\033[1;37;32mR\033[0m'
        return "%s<%s,[%s]>" % (mode, self.function.name,
                                ', '.join(str(i) for i in self))


class IterationFunction(object):

    """A representation of a function describing the direction and the step
    in which an iteration space is traversed."""

    _KNOWN = []

    def __init__(self, name):
        assert isinstance(name, str) and name not in IterationFunction._KNOWN
        self._name = name
        IterationFunction._KNOWN.append(name)

    def __repr__(self):
        return self._name


INC = IterationFunction('++')
"""Unit-stride increment."""

DEC = IterationFunction('--')
"""Unit-stride decrement"""


class TimedAccess(Access):

    """
    A special :class:`Access` object enriched with: ::

        * a "timestamp"; that is, an integer indicating the access location
          within the execution flow;
        * an array of directions; there is one direction for each index,
          indicating whether the index function is monotonically increasing
          or decreasing.
    """

    def __new__(cls, indexed, mode, timestamp):
        assert is_integer(timestamp)
        obj = super(TimedAccess, cls).__new__(cls, indexed, mode)
        obj.timestamp = timestamp
        obj.direction = [DEC if i.reverse else INC for i in obj.findices]
        return obj

    def __eq__(self, other):
        return super(TimedAccess, self).__eq__(other) and\
            isinstance(other, TimedAccess) and\
            self.direction == other.direction

    def __lt__(self, other):
        if not isinstance(other, TimedAccess):
            raise TypeError("Cannot compare with object of type %s" % type(other))
        if self.direction != other.direction:
            raise TypeError("Cannot compare due to mismatching `direction`")
        return super(TimedAccess, self).__lt__(other)

    def lex_eq(self, other):
        return self.timestamp == other.timestamp

    def lex_ne(self, other):
        return self.timestamp != other.timestamp

    def lex_ge(self, other):
        return self.timestamp >= other.timestamp

    def lex_gt(self, other):
        return self.timestamp > other.timestamp

    def lex_le(self, other):
        return self.timestamp <= other.timestamp

    def lex_lt(self, other):
        return self.timestamp < other.timestamp

    def order(self, other):
        if (self.direction != other.direction) or (self.rank != other.rank):
            raise TypeError("Cannot order due to mismatching `direction` and/or `rank`")
        return [i - j if d == INC else j - i
                for i, j, d in zip(self, other, self.direction)]


class Dependence(object):

    """A data dependence between two :class:`Access` objects."""

    def __init__(self, source, sink):
        assert isinstance(source, TimedAccess) and isinstance(sink, TimedAccess)
        assert source.function == sink.function
        self.source = source
        self.sink = sink
        self.findices = source.findices
        self.function = source.function
        self.distance = source.distance(sink)

    @cached_property
    def cause(self):
        """Return the dimension causing the dependence (if any -- return None if
        the dependence is between scalars)."""
        for i, j in zip(self.findices, self.distance):
            try:
                if j > 0:
                    return i
            except TypeError:
                # Conservatively assume this is an offending dimension
                return i

    @cached_property
    def is_indirect(self):
        """Return True if induced by an indirection array (e.g., A[B[i]]),
        False otherwise."""
        for d, i, j in zip(self.findices, self.source.index_mode, self.sink.index_mode):
            if d == self.cause and i == j and i in ['indirect', 'irregular']:
                return True
        return False

    @cached_property
    def is_direct(self):
        """Return True if the dependence occurs through affine functions,
        False otherwise."""
        return not self.is_indirect

    def is_carried(self, dim=None):
        """Return True if a dimension-carried dependence, False otherwise."""
        try:
            return (self.distance > 0) if dim is None else self.cause == dim
        except TypeError:
            # Conservatively assume this is a carried dependence
            return True

    def is_independent(self, dim=None):
        """Return True if a dimension-independent dependence, False otherwise."""
        try:
            return (self.distance == 0) if dim is None else self.cause != dim
        except TypeError:
            # Conservatively assume this is not dimension-independent
            return False

    def is_inplace(self, dim=None):
        """Stronger than ``is_independent()``, as it also compares the timestamps."""
        return self.is_independent() and self.source.lex_eq(self.sink)

    def __repr__(self):
        return "%s -> %s" % (self.source, self.sink)


class DependenceGroup(list):

    @cached_property
    def cause(self):
        ret = [i.cause for i in self if i.cause is not None]
        return tuple(filter_sorted(ret, key=lambda i: i.name))

    @property
    def none(self):
        return len(self) == 0

    def direct(self):
        """Return the dependences induced through affine index functions."""
        return DependenceGroup(i for i in self if i.is_direct)

    def indirect(self):
        """Return the dependences induced through an indirection array."""
        return DependenceGroup(i for i in self if i.is_indirect)

    def carried(self, dim=None):
        """Return the dimension-carried dependences."""
        return DependenceGroup(i for i in self if i.is_carried(dim))

    def independent(self, dim=None):
        """Return the dimension-independent dependences."""
        return DependenceGroup(i for i in self if i.is_independent(dim))

    def inplace(self, dim=None):
        """Return the in-place dependences."""
        return DependenceGroup(i for i in self if i.is_inplace(dim))

    def __add__(self, other):
        assert isinstance(other, DependenceGroup)
        return DependenceGroup(super(DependenceGroup, self).__add__(other))

    def __sub__(self, other):
        assert isinstance(other, DependenceGroup)
        return DependenceGroup([i for i in self if i not in other])


class Scope(object):

    def __init__(self, exprs):
        """
        Initialize a Scope, which represents a group of :class:`Access` objects
        extracted from some expressions ``exprs``. The expressions are to be
        provided as they appear in program order.
        """
        exprs = as_tuple(exprs)
        assert all(isinstance(i, Eq) for i in exprs)

        self.reads = {}
        self.writes = {}
        for i, e in enumerate(exprs):
            # reads
            for j in retrieve_indexed(e.rhs):
                v = self.reads.setdefault(j.base.function, [])
                v.append(TimedAccess(j, 'R', i))
            # write
            if e.lhs.is_Indexed:
                v = self.writes.setdefault(e.lhs.base.function, [])
                v.append(TimedAccess(e.lhs, 'W', i))

    def getreads(self, function):
        return as_tuple(self.reads.get(function))

    def getwrites(self, function):
        return as_tuple(self.writes.get(function))

    def __getitem__(self, function):
        return self.getwrites(function) + self.getreads(function)

    def __repr__(self):
        tracked = filter_sorted(set(self.reads) | set(self.writes), key=lambda i: i.name)
        maxlen = max(1, max([len(i.name) for i in tracked]))
        out = "{:>%d} =>  W : {}\n{:>%d}     R : {}" % (maxlen, maxlen)
        pad = " "*(maxlen + 9)
        reads = [self.getreads(i) for i in tracked]
        for i, r in enumerate(list(reads)):
            if not r:
                reads[i] = ''
                continue
            first = "%s" % tuple.__repr__(r[0])
            shifted = "\n".join("%s%s" % (pad, tuple.__repr__(j)) for j in r[1:])
            shifted = "%s%s" % ("\n" if shifted else "", shifted)
            reads[i] = first + shifted
        writes = [self.getwrites(i) for i in tracked]
        for i, w in enumerate(list(writes)):
            if not w:
                writes[i] = ''
                continue
            first = "%s" % tuple.__repr__(w[0])
            shifted = "\n".join("%s%s" % (pad, tuple.__repr__(j)) for j in w[1:])
            shifted = "%s%s" % ("\n" if shifted else "", shifted)
            writes[i] = '\033[1;37;31m%s\033[0m' % (first + shifted)
        return "\n".join([out.format(i.name, w, '', r)
                          for i, r, w in zip(tracked, reads, writes)])

    @cached_property
    def d_flow(self):
        """Retrieve the flow dependencies, or true dependencies, or read-after-write."""
        found = DependenceGroup()
        for k, v in self.writes.items():
            for w in v:
                for r in self.reads.get(k, []):
                    try:
                        is_flow = (r < w) or (r == w and r.lex_ge(w))
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence
                        is_flow = True
                    if is_flow:
                        found.append(Dependence(w, r))
        return found

    @cached_property
    def d_anti(self):
        """Retrieve the anti dependencies, or write-after-read."""
        found = DependenceGroup()
        for k, v in self.writes.items():
            for w in v:
                for r in self.reads.get(k, []):
                    try:
                        is_anti = (r > w) or (r == w and r.lex_lt(w))
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence
                        is_anti = True
                    if is_anti:
                        found.append(Dependence(r, w))
        return found

    @cached_property
    def d_output(self):
        """Retrieve the output dependencies, or write-after-write."""
        found = DependenceGroup()
        for k, v in self.writes.items():
            for w1 in v:
                for w2 in self.writes.get(k, []):
                    try:
                        is_output = (w2 > w1) or (w2 == w1 and w2.lex_gt(w1))
                    except TypeError:
                        # Non-integer vectors are not comparable.
                        # Conservatively, we assume it is a dependence
                        is_output = True
                    if is_output:
                        found.append(Dependence(w2, w1))
        return found

    @cached_property
    def d_all(self):
        """Retrieve all flow, anti, and output dependences."""
        return self.d_flow + self.d_anti + self.d_output