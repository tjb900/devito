import weakref

import numpy as np
import pytest
from conftest import skipif_yask

from devito import Grid, Function, TimeFunction, Constant, clear_cache
from devito.types import _SymbolCache


@skipif_yask
@pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
def test_cache_function_new(FunctionType):
    """Test that new u[x, y] instances don't cache"""
    grid = Grid(shape=(3, 4))
    u0 = FunctionType(name='u', grid=grid)
    u0.data[:] = 6.
    u1 = FunctionType(name='u', grid=grid)
    u1.data[:] = 2.
    assert np.allclose(u0.data, 6.)
    assert np.allclose(u1.data, 2.)


@skipif_yask
@pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
def test_cache_function_same_indices(FunctionType):
    """Test caching of derived u[x, y] instance from derivative"""
    grid = Grid(shape=(3, 4))
    u0 = FunctionType(name='u', grid=grid)
    u0.data[:] = 6.
    # Pick u(x, y) and u(x + h_x, y) from derivative
    u1 = u0.dx.args[1].args[2]
    u2 = u0.dx.args[0].args[1]
    assert np.allclose(u1.data, 6.)
    assert np.allclose(u2.data, 6.)


@skipif_yask
@pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
def test_cache_function_different_indices(FunctionType):
    """Test caching of u[x + h, y] instance from derivative"""
    grid = Grid(shape=(3, 4))
    u0 = FunctionType(name='u', grid=grid)
    u0.data[:] = 6.
    # Pick u[x + h, y] (different indices) from derivative
    u = u0.dx.args[0].args[1]
    assert np.allclose(u.data, u0.data)


@skipif_yask
def test_cache_constant_new():
    """Test that new u[x, y] instances don't cache"""
    u0 = Constant(name='u')
    u0.data = 6.
    u1 = Constant(name='u')
    u1.data = 2.
    assert u0.data == 6.
    assert u1.data == 2.


@skipif_yask
def test_symbol_cache_aliasing():
    """Test to assert that our aiasing cache isn't defeated by sympys
    non-aliasing symbol cache.

    For further explanation consider the symbol u[x, y] and it's first
    derivative in x, which includes the symbols u[x, y] and u[x + h, y].
    The two functions are aliased in devito's caching mechanism to allow
    multiple stencil indices pointing at the same data object u, but
    SymPy treats these two instances as separate functions and thus is
    allowed to delete one or the other when the cache is cleared.

    The test below asserts that u[x + h, y] is deleted, the data on u
    is still intact through our own caching mechanism."""

    # Ensure a clean cache to start with
    clear_cache()
    # FIXME: Currently not working, presumably due to our
    # failure to cache new instances?
    # assert(len(_SymbolCache) == 0)

    # Create first instance of u and fill its data
    grid = Grid(shape=(3, 4))
    u = Function(name='u', grid=grid)
    u.data[:] = 6.
    u_ref = weakref.ref(u.data)

    # Create u[x + h, y] and delete it again
    dx = u.dx  # Contains two u symbols: u[x, y] and u[x + h, y]
    del dx
    clear_cache()
    # FIXME: Unreliable cache sizes
    # assert len(_SymbolCache) == 1  # We still have a reference to u
    assert np.allclose(u.data, 6.)  # u.data is alive and well

    # Remove the final instance and ensure u.data got deallocated
    del u
    clear_cache()
    assert u_ref() is None


@skipif_yask
def test_symbol_cache_aliasing_reverse():
    """Test to assert that removing he original u[x, y] instance does
    not impede our alisaing cache or leaks memory.
    """

    # Ensure a clean cache to start with
    clear_cache()
    # FIXME: Currently not working, presumably due to our
    # failure to cache new instances?
    # assert(len(_SymbolCache) == 0)

    # Create first instance of u and fill its data
    grid = Grid(shape=(3, 4))
    u = Function(name='u', grid=grid)
    u.data[:] = 6.
    u_ref = weakref.ref(u.data)

    # Create derivative and delete orignal u[x, y]
    dx = u.dx
    del u
    clear_cache()
    # We still have a references to u
    # FIXME: Unreliable cache sizes
    # assert len(_SymbolCache) == 1
    # Ensure u[x + h, y] still holds valid data
    assert np.allclose(dx.args[0].args[1].data, 6.)

    del dx
    clear_cache()
    # FIXME: Unreliable cache sizes
    # assert len(_SymbolCache) == 0  # We still have a reference to u_h
    assert u_ref() is None


@skipif_yask
def test_clear_cache(nx=1000, ny=1000):
    clear_cache()
    grid = Grid(shape=(nx, ny), dtype=np.float64)
    cache_size = len(_SymbolCache)

    for i in range(10):
        assert(len(_SymbolCache) == cache_size)

        Function(name='u', grid=grid, space_order=2)

        assert(len(_SymbolCache) == cache_size + 1)

        clear_cache()


@skipif_yask
def test_cache_after_indexification():
    """Test to assert that the SymPy cache retrieves the right Devito data object
    after indexification.
    """
    grid = Grid(shape=(4, 4, 4))
    u0 = Function(name='u', grid=grid, space_order=0)
    u1 = Function(name='u', grid=grid, space_order=1)
    u2 = Function(name='u', grid=grid, space_order=2)

    for i in [u0, u1, u2]:
        assert i.indexify().base.function.space_order ==\
            (i.indexify() + 1.).args[1].base.function.space_order
