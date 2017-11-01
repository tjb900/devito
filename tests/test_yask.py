from sympy import cos
import numpy as np

import pytest  # noqa

pexpect = pytest.importorskip('yask')  # Run only if YASK is available

from devito import (Eq, Grid, Operator, Constant, Function, TimeFunction,
                    SparseFunction, Backward, configuration, clear_cache)  # noqa
from devito.dle import retrieve_iteration_tree  # noqa
from devito.yask.wrappers import YaskGrid, contexts  # noqa

# For the acoustic wave test
from examples.seismic.acoustic import AcousticWaveSolver, iso_stencil  # noqa
from examples.seismic import demo_model, PointSource, RickerSource, Receiver  # noqa

pytestmark = pytest.mark.skipif(configuration['backend'] != 'yask',
                                reason="'yask' wasn't selected as backend on startup")


def setup_module(module):
    """Get rid of any YASK modules generated and JIT-compiled in previous runs.
    This is not strictly necessary for the tests, but it helps in keeping the
    lib directory clean, which may be helpful for offline analysis."""
    contexts.dump()


@pytest.fixture(scope="module")
def u(dims):
    grid = Grid(shape=(16, 16, 16))
    u = Function(name='yu3D', grid=grid, space_order=0)
    u.data  # Trigger initialization
    return u


@pytest.fixture(autouse=True)
def reset_isa():
    """Force back to NO-SIMD after each test, as some tests may optionally
    switch on SIMD."""
    configuration.yask['develop-mode'] = True


class TestGrids(object):

    """
    Test YASK grid wrappers.
    """

    @classmethod
    def setup_class(cls):
        clear_cache()

    def test_data_type(self, u):
        assert type(u._data_object) == YaskGrid

    @pytest.mark.xfail(reason="YASK always seems to use 3D grids")
    def test_data_movement_1D(self):
        grid = Grid(shape=(16,))
        u = Function(name='yu1D', grid=grid, space_order=0)
        u.data
        assert type(u._data_object) == YaskGrid

        u.data[1] = 1.
        assert u.data[0] == 0.
        assert u.data[1] == 1.
        assert all(i == 0 for i in u.data[2:])

    def test_data_movement_nD(self, u):
        """
        Tests packing/unpacking data to/from YASK through Devito's YaskGrid.
        """
        # Test simple insertion and extraction
        u.data[0, 1, 1] = 1.
        assert u.data[0, 0, 0] == 0.
        assert u.data[0, 1, 1] == 1.
        assert np.all(u.data == u.data[:, :, :])
        assert 1. in u.data[0]
        assert 1. in u.data[0, 1]

        # Test negative indices
        assert u.data[0, -15, -15] == 1.
        u.data[6, 0, 0] = 1.
        assert u.data[-10, :, :].sum() == 1.

        # Test setting whole array to given value
        u.data[:] = 3.
        assert np.all(u.data == 3.)

        # Test insertion of single value into block
        u.data[5, :, 5] = 5.
        assert np.all(u.data[5, :, 5] == 5.)

        # Test extraction of block with negative indices
        sliced = u.data[-11, :, -11]
        assert sliced.shape == (16,)
        assert np.all(sliced == 5.)

        # Test insertion of block into block
        block = np.ndarray(shape=(1, 16, 1), dtype=np.float32)
        block.fill(4.)
        u.data[4, :, 4] = block
        assert np.all(u.data[4, :, 4] == block)

    def test_data_arithmetic_nD(self, u):
        """
        Tests arithmetic operations through YaskGrids.
        """
        u.data[:] = 1

        # Simple arithmetic
        assert np.all(u.data == 1)
        assert np.all(u.data + 2. == 3.)
        assert np.all(u.data - 2. == -1.)
        assert np.all(u.data * 2. == 2.)
        assert np.all(u.data / 2. == 0.5)
        assert np.all(u.data % 2 == 1.)

        # Increments and partial increments
        u.data[:] += 2.
        assert np.all(u.data == 3.)
        u.data[9, :, :] += 1.
        assert all(np.all(u.data[i, :, :] == 3.) for i in range(9))
        assert np.all(u.data[9, :, :] == 4.)

        # Right operations __rOP__
        u.data[:] = 1.
        arr = np.ndarray(shape=(16, 16, 16), dtype=np.float32)
        arr.fill(2.)
        assert np.all(arr - u.data == -1.)


class TestOperatorSimple(object):

    """
    Test execution of "toy" Operators through YASK.
    """

    @classmethod
    def setup_class(cls):
        clear_cache()

    @pytest.mark.parametrize("space_order", [0, 1, 2])
    @pytest.mark.parametrize("nosimd", [True, False])
    def test_increasing_halo_wo_ofs(self, space_order, nosimd):
        """
        Apply the trivial equation ``u[t+1,x,y,z] = u[t,x,y,z] + 1`` and check
        that increasing space orders lead to proportionately larger halo regions,
        which are *not* written by the Operator.
        For example, with ``space_order = 0``, produce (in 2D view):

            1 1 1 ... 1 1
            1 1 1 ... 1 1
            1 1 1 ... 1 1
            1 1 1 ... 1 1
            1 1 1 ... 1 1

        With ``space_order = 1``, produce:

            0 0 0 0 0 0 0 0 0
            0 1 1 1 ... 1 1 0
            0 1 1 1 ... 1 1 0
            0 1 1 1 ... 1 1 0
            0 1 1 1 ... 1 1 0
            0 1 1 1 ... 1 1 0
            0 0 0 0 0 0 0 0 0

        And so on and so forth.
        """
        # SIMD on/off
        configuration.yask['develop-mode'] = nosimd

        grid = Grid(shape=(16, 16, 16))
        u = TimeFunction(name='yu4D', grid=grid, space_order=space_order)
        u.data.with_halo[:] = 0.
        op = Operator(Eq(u.forward, u + 1.))
        op(yu4D=u, t=1)
        assert 'run_solution' in str(op)
        # Chech that the domain size has actually been written to
        assert np.all(u.data[1] == 1.)
        # Check that the halo planes are still 0
        assert all(np.all(u.data.with_halo[1, i, :, :] == 0) for i in range(space_order))
        assert all(np.all(u.data.with_halo[1, :, i, :] == 0) for i in range(space_order))
        assert all(np.all(u.data.with_halo[1, :, :, i] == 0) for i in range(space_order))

    def test_increasing_multi_steps(self):
        """
        Apply the trivial equation ``u[t+1,x,y,z] = u[t,x,y,z] + 1`` for 11
        timesteps and check that all grid domain values are equal to 11 within
        ``u[1]`` and equal to 10 within ``u[0]``.
        """
        grid = Grid(shape=(8, 8, 8))
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        u.data.with_halo[:] = 0.
        op = Operator(Eq(u.forward, u + 1.))
        op(yu4D=u, t=11)
        assert 'run_solution' in str(op)
        assert np.all(u.data[0] == 10.)
        assert np.all(u.data[1] == 11.)

    @pytest.mark.parametrize("space_order", [2])
    def test_fixed_halo_w_ofs(self, space_order):
        """
        Compute an N-point stencil sum, where N is the number of points sorrounding
        an inner (i.e., non-border) grid point.
        For example (in 2D view):

            1 1 1 ... 1 1
            1 4 4 ... 4 1
            1 4 4 ... 4 1
            1 4 4 ... 4 1
            1 1 1 ... 1 1
        """
        grid = Grid(shape=(16, 16, 16))
        v = TimeFunction(name='yv4D', grid=grid, space_order=space_order)
        v.data.with_halo[:] = 1.
        op = Operator(Eq(v.forward, v.laplace + 6*v), subs=grid.spacing_map)
        op(yv4D=v, t=1)
        assert 'run_solution' in str(op)
        # Chech that the domain size has actually been written to
        assert np.all(v.data[1] == 6.)
        # Check that the halo planes are untouched
        assert all(np.all(v.data.with_halo[1, i, :, :] == 1) for i in range(space_order))
        assert all(np.all(v.data.with_halo[1, :, i, :] == 1) for i in range(space_order))
        assert all(np.all(v.data.with_halo[1, :, :, i] == 1) for i in range(space_order))

    def test_mixed_space_order(self):
        """
        Make sure that no matter whether data objects have different space order,
        as long as they have same domain, the Operator will be executed correctly.
        """
        grid = Grid(shape=(8, 8, 8))
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        v = TimeFunction(name='yv4D', grid=grid, space_order=1)
        u.data.with_halo[:] = 1.
        v.data.with_halo[:] = 2.
        op = Operator(Eq(v.forward, u + v))
        op(yu4D=u, yv4D=v, t=1)
        assert 'run_solution' in str(op)
        # Chech that the domain size has actually been written to
        assert np.all(v.data[1] == 3.)
        # Check that the halo planes are untouched
        assert np.all(v.data.with_halo[1, 0, :, :] == 2)
        assert np.all(v.data.with_halo[1, :, 0, :] == 2)
        assert np.all(v.data.with_halo[1, :, :, 0] == 2)

    def test_multiple_loop_nests(self):
        """
        Compute a simple stencil S, preceded by an "initialization loop" I and
        followed by a "random loop" R.

            * S is the trivial equation ``u[t+1,x,y,z] = u[t,x,y,z] + 1``;
            * I initializes ``u`` to 0;
            * R adds 2 to another field ``v`` along the ``z`` dimension but only
                over the planes ``[x=0, y=2]`` and ``[x=0, y=5]``.

        Out of these three loop nests, only S should be "offloaded" to YASK; indeed,
        I is outside the time loop, while R does not loop over space dimensions.
        This test checks that S is the only loop nest "offloaded" to YASK, and
        that the numerical output is correct.
        """
        grid = Grid(shape=(12, 12, 12))
        x, y, z = grid.dimensions
        t = grid.stepping_dim
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        v = TimeFunction(name='yv4D', grid=grid, space_order=0)
        v.data[:] = 0.
        eqs = [Eq(u.indexed[0, x, y, z], 0),
               Eq(u.indexed[1, x, y, z], 0),
               Eq(u.forward, u + 1.),
               Eq(v.indexed[t + 1, 0, 2, z], v.indexed[t + 1, 0, 2, z] + 2.),
               Eq(v.indexed[t + 1, 0, 5, z], v.indexed[t + 1, 0, 5, z] + 2.)]
        op = Operator(eqs)
        op(yu4D=u, yv4D=v, t=1)
        assert 'run_solution' in str(op)
        assert len(retrieve_iteration_tree(op)) == 3
        assert np.all(u.data[0] == 0.)
        assert np.all(u.data[1] == 1.)
        assert np.all(v.data[0] == 0.)
        assert np.all(v.data[1, 0, 2] == 2.)
        assert np.all(v.data[1, 0, 5] == 2.)

    def test_irregular_write(self):
        """
        Compute a simple stencil S w/o offloading it to YASK because of the presence
        of indirect write accesses (e.g. A[B[i]] = ...); YASK grid functions are however
        used in the generated code to access the data at the right location. This
        test checks that the numerical output is correct after this transformation.

        Initially, the input array (a YASK grid, under the hood), at t=0 is (2D view):

            0 1 2 3
            0 1 2 3
            0 1 2 3
            0 1 2 3

        Then, the Operator "flips" its content, and at timestep t=1 we get (2D view):

            3 2 1 0
            3 2 1 0
            3 2 1 0
            3 2 1 0
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        t = grid.stepping_dim
        p = SparseFunction(name='points', grid=grid, nt=1, npoint=4)
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u.data[0, i, j, k] = k
        ind = lambda i: p.indexed[0, i]
        eqs = [Eq(p.indexed[0, 0], 3.), Eq(p.indexed[0, 1], 2.),
               Eq(p.indexed[0, 2], 1.), Eq(p.indexed[0, 3], 0.),
               Eq(u.indexed[t + 1, ind(x), ind(y), ind(z)], u.indexed[t, x, y, z])]
        op = Operator(eqs, subs=grid.spacing_map)
        op(yu4D=u, time=1)
        assert 'run_solution' not in str(op)
        assert all(np.all(u.data[1, :, :, i] == 3 - i) for i in range(4))

    def test_reverse_time_loop(self):
        """
        Check that YASK evaluates stencil equations correctly when iterating in the
        reverse time direction.
        """
        grid = Grid(shape=(4, 4, 4))
        u = TimeFunction(name='yu4D', grid=grid, space_order=0, time_order=2)
        u.data[:] = 2.
        eq = Eq(u.backward, u - 1.)
        op = Operator(eq, time_axis=Backward)
        op(yu4D=u, t=2)
        assert 'run_solution' in str(op)
        assert np.all(u.data[2] == 2.)
        assert np.all(u.data[1] == 1.)
        assert np.all(u.data[0] == 0.)

    def test_capture_vector_temporaries(self):
        """
        Check that all vector temporaries appearing in a offloaded stencil
        equation are: ::

            * mapped to a YASK grid, directly in Python-land,
            * so no memory needs to be allocated in C-land, and
            * passed down to the generated code, and
            * re-initializaed to 0. at each operator application
        """
        grid = Grid(shape=(4, 4, 4))
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        v = Function(name='yv3D', grid=grid, space_order=0)
        eqs = [Eq(u.forward, u + cos(v)*2. + cos(v)*cos(v)*3.)]
        op = Operator(eqs)
        # Sanity check of the generated code
        assert 'posix_memalign' not in str(op)
        assert 'run_solution' in str(op)
        # No data has been allocated for the temporaries yet
        assert op.yk_soln.grids['r0'].is_storage_allocated() is False
        op.apply(yu4D=u, yv3D=v, t=1)
        # Temporary data has already been released after execution
        assert op.yk_soln.grids['r0'].is_storage_allocated() is False
        assert np.all(v.data == 0.)
        assert np.all(u.data[1] == 5.)

    def test_constants(self):
        """
        Check that :class:`Constant` objects are treated correctly.
        """
        grid = Grid(shape=(4, 4, 4))
        c = Constant(name='c', value=2.)
        p = SparseFunction(name='points', grid=grid, nt=1, npoint=1)
        u = TimeFunction(name='yu4D', grid=grid, space_order=0)
        u.data[:] = 0.
        op = Operator([Eq(u.forward, u + c), Eq(p.indexed[0, 0], 1. + c)])
        assert 'run_solution' in str(op)
        op.apply(yu4D=u, c=c, t=11)
        # Check YASK did its job and could read constant grids w/o problems
        assert np.all(u.data[0] == 20.)
        # Check the Constant could be read correctly even in Devito-land, i.e.,
        # outside of run_solution
        assert p.data[0][0] == 3.
        # Check re-executing with another constant gives the correct result
        c2 = Constant(name='c', value=5.)
        op.apply(yu4D=u, c=c2, t=3)
        assert np.all(u.data[0] == 30.)
        assert p.data[0][0] == 6.


class TestOperatorAcoustic(object):

    """
    Test the acoustic wave model through YASK.

    This test is very similar to the one in test_adjointA.
    """

    @pytest.fixture
    def shape(self):
        return (60, 70, 80)

    @pytest.fixture
    def nbpml(self):
        return 10

    @pytest.fixture
    def model(self, shape, nbpml):
        return demo_model(spacing=[15., 15., 15.], shape=shape, nbpml=nbpml,
                          preset='layers-isotropic', ratio=3)

    @pytest.fixture
    def time_params(self, model):
        # Derive timestepping from model spacing
        t0 = 0.0  # Start time
        tn = 500.  # Final time
        dt = model.critical_dt
        nt = int(1 + (tn-t0) / dt)  # Number of timesteps
        return t0, tn, nt

    @pytest.fixture
    def m(self, model):
        return model.m

    @pytest.fixture
    def damp(self, model):
        return model.damp

    @pytest.fixture
    def space_order(self):
        return 4

    @pytest.fixture
    def time_order(self):
        return 2

    @pytest.fixture
    def u(self, model, space_order, time_order):
        return TimeFunction(name='u', grid=model.grid,
                            space_order=space_order, time_order=time_order)

    @pytest.fixture
    def eqn(self, m, damp, u, time_order):
        t = u.grid.stepping_dim
        return iso_stencil(u, time_order, m, t.spacing, damp)

    @pytest.fixture
    def src(self, model, time_params):
        time_values = np.linspace(*time_params)  # Discretized time axis
        # Define source geometry (center of domain, just below surface)
        src = RickerSource(name='src', grid=model.grid, f0=0.01, time=time_values)
        src.coordinates.data[0, :] = np.array(model.domain_size) * .5
        src.coordinates.data[0, -1] = 30.
        return src

    @pytest.fixture
    def rec(self, model, time_params, src):
        nrec = 130  # Number of receivers
        t0, tn, nt = time_params
        rec = Receiver(name='rec', grid=model.grid, ntime=nt, npoint=nrec)
        rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
        rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]
        return rec

    def test_acoustic_wo_src_wo_rec(self, model, eqn, m, damp, u):
        """
        Test that the acoustic wave equation runs without crashing in absence
        of sources and receivers.
        """
        dt = model.critical_dt
        u.data[:] = 0.0
        op = Operator(eqn, subs=model.spacing_map)
        assert 'run_solution' in str(op)

        op.apply(u=u, m=m, damp=damp, t=10, dt=dt)

    @pytest.mark.xfail
    def test_acoustic_w_src_wo_rec(self, model, eqn, m, damp, u, src):
        """
        Test that the acoustic wave equation runs without crashing in absence
        of receivers.
        """
        dt = model.critical_dt
        u.data[:] = 0.0
        eqns = eqn
        eqns += src.inject(field=u.forward, expr=src * dt**2 / m, offset=model.nbpml)
        op = Operator(eqns, subs=model.spacing_map)
        assert 'run_solution' in str(op)

        op.apply(u=u, m=m, damp=damp, src=src, t=1, dt=dt)

        exp_u = 152.76
        assert np.isclose(np.linalg.norm(u.data[:]), exp_u, atol=exp_u*1.e-2)

    @pytest.mark.xfail
    def test_acoustic_w_src_w_rec(self, model, eqn, m, damp, u, src, rec):
        """
        Test that the acoustic wave equation forward operator produces the correct
        results when running a 3D model also used in ``test_adjointA.py``.
        """
        dt = model.critical_dt
        u.data[:] = 0.0
        eqns = eqn
        eqns += src.inject(field=u.forward, expr=src * dt**2 / m, offset=model.nbpml)
        eqns += rec.interpolate(expr=u, offset=model.nbpml)
        op = Operator(eqns, subs=model.spacing_map)
        assert 'run_solution' in str(op)

        op.apply(u=u, m=m, damp=damp, src=src, rec=rec, t=1, dt=dt)

        # The expected norms have been computed "by hand" looking at the output
        # of test_adjointA's forward operator w/o using the YASK backend.
        exp_u = 152.76
        exp_rec = 212.00
        assert np.isclose(np.linalg.norm(u.data[:]), exp_u, atol=exp_u*1.e-2)
        assert np.isclose(np.linalg.norm(rec.data), exp_rec, atol=exp_rec*1.e-2)

    def test_acoustic_adjoint(self, shape, time_order, space_order, nbpml):
        """
        Full acoustic wave test, forward + adjoint operators
        """
        from test_adjointA import test_acoustic
        test_acoustic('layers', shape, time_order, space_order, nbpml)
