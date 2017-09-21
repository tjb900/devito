# python $DEVITO_HOME/examples/benchmark.py plot -P $problem -a -o -d $grid $grid $grid -so $space_orders -to $time_orders -r $DEVITO_RESULTS -p $DEVITO_PLOTS --max_bw 18.2 --max_flops 210

import os
import sys
from subprocess import call

if len(sys.argv) != 2:
    print "Usage: python plot.py machine=[endeavour,hero]"

machines = {}
execfile('machines.txt', machines)

devito_home = os.environ['DEVITO_HOME']
results_dir = os.path.join(devito_home, '..', 'experimentation')
raw_dir = [x[0] for x in os.walk(os.path.join(results_dir, 'raw', sys.argv[1]))]
plots_dir = os.path.join(results_dir, 'plots', sys.argv[1])

for i in raw_dir[1:]:
    location = i.split('/')[-1]
    problem, mode, grid, arch, backend = location.split('-')
    files = [f for f in os.listdir(i) if f.endswith('json')]
    if not files:
        print "WTF? "
        sys.exit(0)
    sample = files[0]

    if arch not in machines:
        print "Don't own any information for machine", arch
        sys.exit(0)
    machine = machines[arch]

    grid = sample[sample.find("(")+1:sample.find(")")].split(',')[0]
    space_orders = set(f[f.find('space_order')+11:].split('_')[0] for f in files)
    space_orders = str(sorted([int(j) for j in list(space_orders)]))
    space_orders = space_orders.replace(',', '').replace('[', '').replace(']', '')
    tn = sample.split('tn')[1].split('.')[0]

    if 'ekf' in arch:
        os.environ['DEVITO_ARCH'] = 'knl'
    else:
        os.environ['DEVITO_ARCH'] = 'intel'

    args = ['python', 'benchmark.py', '--bench-mode', mode, 'plot', '-P', problem, '-a', '-d',
            grid, grid, grid, '-so', space_orders, '-to', '2', '--tn', tn,
            '-r', i, '-p', plots_dir,
            '--max_bw', str(machine['dram-stream-bw']),
            '--max_flops', str(machine['machine-peak']),
            '--point_runtime',
            '--arch', arch]
    call(' '.join(args), shell=True)
