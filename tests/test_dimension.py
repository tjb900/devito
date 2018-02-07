import numpy as np

from conftest import skipif_yask

from devito import ConditionalDimension, Grid, TimeFunction, Eq, Operator


@skipif_yask
def test_conditional():
    nt = 16
    grid = Grid(shape=(11, 11))
    time = grid.time_dim

    u = TimeFunction(name='u', grid=grid)
    assert(grid.stepping_dim in u.indices)

    u2 = TimeFunction(name='u2', grid=grid, save=nt)
    assert(time in u2.indices)

    factor = 4
    time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
    usave = TimeFunction(name='usave', grid=grid, save=factor, time_dim=time_subsampled)
    assert(time_subsampled in usave.indices)

    eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.), Eq(usave, u)]
    op = Operator(eqns)
    op.apply(t=nt)
    assert np.all(u.data[1] == 15)
    assert np.all(u2.data[i] == i for i in range(nt))
    assert np.all(usave.data[i] == i*factor for i in range(factor))
