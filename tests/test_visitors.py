import cgen as c
import pytest
from conftest import skipif_yask

from devito import Eq
from devito.ir.iet import (Block, Expression, Callable, FindSections,
                           FindSymbols, IsPerfectIteration, MergeOuterIterations,
                           Transformer, NestedTransformer, printAST)


@pytest.fixture(scope="module")
def exprs(a, b):
    return [Expression(Eq(a, a + b + 5.)),
            Expression(Eq(a, b - a)),
            Expression(Eq(a, 4 * (b * a))),
            Expression(Eq(a, (6. / b) + (8. * a)))]


@pytest.fixture(scope="module")
def block1(exprs, iters):
    # Perfect loop nest:
    # for i
    #   for j
    #     for k
    #       expr0
    return iters[0](iters[1](iters[2](exprs[0])))


@pytest.fixture(scope="module")
def block2(exprs, iters):
    # Non-perfect simple loop nest:
    # for i
    #   expr0
    #   for j
    #     for k
    #       expr1
    return iters[0]([exprs[0], iters[1](iters[2](exprs[1]))])


@pytest.fixture(scope="module")
def block3(exprs, iters):
    # Non-perfect non-trivial loop nest:
    # for i
    #   for s
    #     expr0
    #   for j
    #     for k
    #       expr1
    #       expr2
    #   for p
    #     expr3
    return iters[0]([iters[3](exprs[0]),
                     iters[1](iters[2]([exprs[1], exprs[2]])),
                     iters[4](exprs[3])])


@skipif_yask
def test_printAST(block1, block2, block3):
    str1 = printAST(block1)
    assert str1 in """
<Iteration i::i::[0, 3, 1]::[0, 0]>
  <Iteration j::j::[0, 5, 1]::[0, 0]>
    <Iteration k::k::[0, 7, 1]::[0, 0]>
      <Expression a[i] = a[i] + b[i] + 5.0>
"""

    str2 = printAST(block2)
    assert str2 in """
<Iteration i::i::[0, 3, 1]::[0, 0]>
  <Expression a[i] = a[i] + b[i] + 5.0>
  <Iteration j::j::[0, 5, 1]::[0, 0]>
    <Iteration k::k::[0, 7, 1]::[0, 0]>
      <Expression a[i] = -a[i] + b[i]>
"""

    str3 = printAST(block3)
    assert str3 in """
<Iteration i::i::[0, 3, 1]::[0, 0]>
  <Iteration s::s::[0, 4, 1]::[0, 0]>
    <Expression a[i] = a[i] + b[i] + 5.0>
  <Iteration j::j::[0, 5, 1]::[0, 0]>
    <Iteration k::k::[0, 7, 1]::[0, 0]>
      <Expression a[i] = -a[i] + b[i]>
      <Expression a[i] = 4*a[i]*b[i]>
  <Iteration q::q::[0, 4, 1]::[0, 0]>
    <Expression a[i] = 8.0*a[i] + 6.0/b[i]>
"""


@skipif_yask
def test_create_cgen_tree(block1, block2, block3):
    assert str(Callable('foo', block1, 'void', ()).ccode) == """\
void foo()
{
  for (int i = 0; i < 3; i += 1)
  {
    for (int j = 0; j < 5; j += 1)
    {
      for (int k = 0; k < 7; k += 1)
      {
        a[i] = a[i] + b[i] + 5.0F;
      }
    }
  }
}"""
    assert str(Callable('foo', block2, 'void', ()).ccode) == """\
void foo()
{
  for (int i = 0; i < 3; i += 1)
  {
    a[i] = a[i] + b[i] + 5.0F;
    for (int j = 0; j < 5; j += 1)
    {
      for (int k = 0; k < 7; k += 1)
      {
        a[i] = -a[i] + b[i];
      }
    }
  }
}"""
    assert str(Callable('foo', block3, 'void', ()).ccode) == """\
void foo()
{
  for (int i = 0; i < 3; i += 1)
  {
    for (int s = 0; s < 4; s += 1)
    {
      a[i] = a[i] + b[i] + 5.0F;
    }
    for (int j = 0; j < 5; j += 1)
    {
      for (int k = 0; k < 7; k += 1)
      {
        a[i] = -a[i] + b[i];
        a[i] = 4*a[i]*b[i];
      }
    }
    for (int q = 0; q < 4; q += 1)
    {
      a[i] = 8.0F*a[i] + 6.0F/b[i];
    }
  }
}"""


@skipif_yask
def test_find_sections(exprs, block1, block2, block3):
    finder = FindSections()

    sections = finder.visit(block1)
    assert len(sections) == 1

    sections = finder.visit(block2)
    assert len(sections) == 2
    found = list(sections.values())
    assert len(found[0]) == 1
    assert found[0][0].stencil == exprs[0].stencil
    assert len(found[1]) == 1
    assert found[1][0].stencil == exprs[1].stencil

    sections = finder.visit(block3)
    assert len(sections) == 3
    found = list(sections.values())
    assert len(found[0]) == 1
    assert found[0][0].stencil == exprs[0].stencil
    assert len(found[1]) == 2
    assert found[1][0].stencil == exprs[1].stencil
    assert found[1][1].stencil == exprs[2].stencil
    assert len(found[2]) == 1
    assert found[2][0].stencil == exprs[3].stencil


@skipif_yask
def test_is_perfect_iteration(block1, block2, block3):
    checker = IsPerfectIteration()

    assert checker.visit(block1) is True
    assert checker.visit(block1.nodes[0]) is True
    assert checker.visit(block1.nodes[0].nodes[0]) is True

    assert checker.visit(block2) is False
    assert checker.visit(block2.nodes[1]) is True
    assert checker.visit(block2.nodes[1].nodes[0]) is True

    assert checker.visit(block3) is False
    assert checker.visit(block3.nodes[0]) is True
    assert checker.visit(block3.nodes[1]) is True
    assert checker.visit(block3.nodes[2]) is True


@skipif_yask
def test_transformer_wrap(exprs, block1, block2, block3):
    """Basic transformer test that wraps an expression in comments"""
    line1 = '// This is the opening comment'
    line2 = '// This is the closing comment'
    wrapper = lambda n: Block(c.Line(line1), n, c.Line(line2))
    transformer = Transformer({exprs[0]: wrapper(exprs[0])})

    for block in [block1, block2, block3]:
        newblock = transformer.visit(block)
        newcode = str(newblock.ccode)
        oldnumlines = len(str(block.ccode).split('\n'))
        newnumlines = len(newcode.split('\n'))
        assert newnumlines >= oldnumlines + 2
        assert line1 in newcode
        assert line2 in newcode
        assert "a[i] = a[i] + b[i] + 5.0F;" in newcode


@skipif_yask
def test_transformer_replace(exprs, block1, block2, block3):
    """Basic transformer test that replaces an expression"""
    line1 = '// Replaced expression'
    replacer = Block(c.Line(line1))
    transformer = Transformer({exprs[0]: replacer})

    for block in [block1, block2, block3]:
        newblock = transformer.visit(block)
        newcode = str(newblock.ccode)
        oldnumlines = len(str(block.ccode).split('\n'))
        newnumlines = len(newcode.split('\n'))
        assert newnumlines >= oldnumlines
        assert line1 in newcode
        assert "a[i0] = a[i0] + b[i0] + 5.0F;" not in newcode


@skipif_yask
def test_transformer_replace_function_body(block1, block2):
    """Create a Function and replace its body with another."""
    args = FindSymbols().visit(block1)
    f = Callable('foo', block1, 'void', args)
    assert str(f.ccode) == """void foo()
{
  for (int i = 0; i < 3; i += 1)
  {
    for (int j = 0; j < 5; j += 1)
    {
      for (int k = 0; k < 7; k += 1)
      {
        a[i] = a[i] + b[i] + 5.0F;
      }
    }
  }
}"""

    f = Transformer({block1: block2}).visit(f)
    assert str(f.ccode) == """void foo()
{
  for (int i = 0; i < 3; i += 1)
  {
    a[i] = a[i] + b[i] + 5.0F;
    for (int j = 0; j < 5; j += 1)
    {
      for (int k = 0; k < 7; k += 1)
      {
        a[i] = -a[i] + b[i];
      }
    }
  }
}"""


@skipif_yask
def test_transformer_add_replace(exprs, block2, block3):
    """Basic transformer test that adds one expression and replaces another"""
    line1 = '// Replaced expression'
    line2 = '// Adding a simple line'
    replacer = Block(c.Line(line1))
    adder = lambda n: Block(c.Line(line2), n)
    transformer = Transformer({exprs[0]: replacer,
                               exprs[1]: adder(exprs[1])})

    for block in [block2, block3]:
        newblock = transformer.visit(block)
        newcode = str(newblock.ccode)
        oldnumlines = len(str(block.ccode).split('\n'))
        newnumlines = len(newcode.split('\n'))
        assert newnumlines >= oldnumlines + 1
        assert line1 in newcode
        assert line2 in newcode
        assert "a[i0] = a[i0] + b[i0] + 5.0F;" not in newcode


@skipif_yask
def test_nested_transformer(exprs, iters, block2):
    """Unlike Transformer, based on BFS, a NestedTransformer applies transformations
    performing a DFS. This test simultaneously replace an inner expression and an
    Iteration sorrounding it."""
    target_loop = block2.nodes[1]
    target_expr = target_loop.nodes[0].nodes[0]
    mapper = {target_loop: iters[3](target_loop.nodes[0]),
              target_expr: exprs[3]}
    processed = NestedTransformer(mapper).visit(block2)
    assert printAST(processed) == """<Iteration i::i::[0, 3, 1]::[0, 0]>
  <Expression a[i] = a[i] + b[i] + 5.0>
  <Iteration s::s::[0, 4, 1]::[0, 0]>
    <Iteration k::k::[0, 7, 1]::[0, 0]>
      <Expression a[i] = 8.0*a[i] + 6.0/b[i]>"""


@skipif_yask
def test_merge_iterations_flat(exprs, iters):
    """Test outer loop merging on a simple two-level hierarchy:

    for i                       for i
        for j              \        for j
            expr0       === \           expr0
    for i               === /       for k
        for k              /            expr1
            expr1
    """
    block = [iters[0](iters[1](exprs[0])),
             iters[0](iters[2](exprs[1]))]
    newblock = MergeOuterIterations().visit(block)
    newstr = printAST(newblock)
    assert newstr == """<Iteration i::i::[0, 3, 1]::[0, 0]>
  <Iteration j::j::[0, 5, 1]::[0, 0]>
    <Expression a[i] = a[i] + b[i] + 5.0>
  <Iteration k::k::[0, 7, 1]::[0, 0]>
    <Expression a[i] = -a[i] + b[i]>"""


@skipif_yask
def test_merge_iterations_deep(exprs, iters):
    """Test outer loop merging on a deep hierarchy:

    for i                       for i
        for j                       for j
            expr0           \           expr0
    for i                === \      for k
        for k            === /          expr0
            expr0           /           expr1
        for k
            expr1
    """
    block = [iters[0](iters[1](exprs[0])),
             iters[0]([iters[2](exprs[0]), iters[2](exprs[1])])]
    newblock = MergeOuterIterations().visit(block)
    newstr = printAST(newblock)
    assert newstr == """<Iteration i::i::[0, 3, 1]::[0, 0]>
  <Iteration j::j::[0, 5, 1]::[0, 0]>
    <Expression a[i] = a[i] + b[i] + 5.0>
  <Iteration k::k::[0, 7, 1]::[0, 0]>
    <Expression a[i] = a[i] + b[i] + 5.0>
    <Expression a[i] = -a[i] + b[i]>"""


@skipif_yask
def test_merge_iterations_nested(exprs, iters):
    """Test outer loop merging on a nested hierarchy that only exposes
    the second-level merge after the first level has been performed:

    for i                       for i
        for j                       for j
            expr0           \           expr0
    for i                === \          expr1
        for j            === /      for k
            expr1           /           expr1
        for k
            expr1
    """
    block = [iters[0](iters[1](exprs[0])),
             iters[0]([iters[1](exprs[1]), iters[2](exprs[1])])]
    newblock = MergeOuterIterations().visit(block)
    newstr = printAST(newblock)
    assert newstr == """<Iteration i::i::[0, 3, 1]::[0, 0]>
  <Iteration j::j::[0, 5, 1]::[0, 0]>
    <Expression a[i] = a[i] + b[i] + 5.0>
    <Expression a[i] = -a[i] + b[i]>
  <Iteration k::k::[0, 7, 1]::[0, 0]>
    <Expression a[i] = -a[i] + b[i]>"""
