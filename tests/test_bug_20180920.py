def test_bug_20180920():
    from devito import (Grid, Function, TimeFunction,
                        Constant, Eq, Operator, Inc)
    import numpy as np

    grid = Grid(shape=(20, 20, 20))
    u = TimeFunction(name='u', grid=grid, space_order=4,
                     time_order=2, save=None)
    usave = TimeFunction(name='usave', grid=grid, space_order=0,
                         time_order=0, save=10)

    g = Function(name='g', grid=grid, space_order=0)
    i = Function(name='i', grid=grid, space_order=0)

    save_shift = Constant(name='save_shift', dtype=np.int32)

    step = Eq(u.forward, u - u.backward + 1)
    g_inc = Inc(g, u * usave.subs(grid.time_dim, grid.time_dim - save_shift))
    i_inc = Inc(i, (usave*usave).subs(grid.time_dim,
                                      grid.time_dim - save_shift))

    op = Operator([step, g_inc, i_inc])

