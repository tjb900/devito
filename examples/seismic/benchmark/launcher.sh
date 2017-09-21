#!/bin/bash

# Launch as:
#
#    problem=[acoustic,tti] order=int grid=int \
#    dse=[...] dle=[...] at=[True,False (default)] mode=[srun,maxperf,dse,dle] \
#    proc=[bdwb,skl,ekf_{flat,cache}] system=[hero,edv] hyperthreads=int \
#    compiler=[gcc,icc,knl] vtune=[...] advisor=[...] cprofile=[...] \
#    ./path/to/devito/examples/launcher.sh
#

if [ -z ${DEVITO_HOME+x} ]; then
    echo "Please, set DEVITO_HOME to the root Devito directory"
    exit
fi

if [ -z ${DEVITO_OUTPUT+x} ]; then
    echo "Please, set DEVITO_OUTPUT to the root results directory"
    exit
fi

if [ -z ${proc+x} ]; then
    proc="bdwb"
fi

if [ -z ${system+x} ]; then
    system="hero"
fi

if [ -z ${grid+x} ]; then
    grid=256
fi

if [ -z ${order+x} ]; then
    order=4
fi

if [ -z ${compiler+x} ]; then
    compiler=intel
fi

if [ "$at" == "True" ]; then
    at="-a"
fi

if [ -n "$dse" ]; then
    dse="-dse "$dse
fi

if [ -n "$dle" ]; then
    dle="-dle "$dle
fi

if [ -z "$hyperthreads" ]; then
    hyperthreads=1
fi

time_orders="2"
if [ "$problem" == "acoustic" ]; then
    space_orders="2 4 6 8 10 12 14 16"
elif [ "$problem" == "tti" ]; then
    space_orders="4 8"
else
    echo "Unrecognised problem $problem (allowed: acoustic, tti)"
    exit
fi

if [ "$system" == "edv" ]; then
    # Need to fetch the compilers on Endeavour
    . /opt/intel/ics/2017.1.132/compilers_and_libraries_2017/linux/bin/compilervars.sh intel64
    . /opt/intel/ics/2017.1.132/vtune_amplifier_xe/amplxe-vars.sh
    . /opt/intel/ics/2017.1.132/advisor_2017/advixe-vars.sh
fi

# Compiler used by Devito
export DEVITO_ARCH=$compiler

# OpenMP always activated when DLE is in advanced mode
export DEVITO_OPENMP=1

# The architecture on which we're running
arch=$system"_"$proc

# Machine-dependent setup
if [ "$arch" == "hero_bdwb" ]; then
    # Hero Broadwell
    export OMP_NUM_THREADS=8
    export KMP_HW_SUBSET=8c,1t
    NUMACTL="numactl --cpubind=0 --membind=0"
elif [ "$arch" == "edv_bdwb" ]; then
    # Endeavour Broadwell
    export OMP_NUM_THREADS=18
    export KMP_HW_SUBSETS=18c,1t
    NUMACTL="numactl --cpubind=0 --membind=0"
elif [ "$arch" == "edv_skl" ]; then
    # Endeavour Skylake
    export OMP_NUM_THREADS=20
    export KMP_HW_SUBSETS=20c,1t
    NUMACTL="numactl --cpubind=0 --membind=0"
elif [ "$arch" == "edv_ekf_flat" ]; then
    # Endeavour KNL in FLAT mode
    export OMP_NUM_THREADS=$[68*${hyperthreads}]
    export KMP_HW_SUBSETS=68c,${hyperthreads}t
    arch=$arch"_"$hyperthreads
    NUMACTL="numactl --cpubind=0 --membind=1"
elif [ "$arch" == "edv_ekf_cache" ]; then
    # Endeavour KNL in CACHE mode
    export OMP_NUM_THREADS=$[68*${hyperthreads}]
    export KMP_HW_SUBSETS=68c,${hyperthreads}t
    arch=$arch"_"$hyperthreads
    NUMACTL="numactl --cpubind=0 --membind=0"
else
    echo "Unrecognized architecture: $arch"
    exit
fi

# Thread affinity tweaks (only intel compiler)
if [ "$hyperthreads" -gt 2 ]; then
    export KMP_AFFINITY=compact
else
    export KMP_AFFINITY=scatter
fi

# Problem name identifier
name=$problem-$mode-grid$grid

# Unique timestep for this run
timestamp=$(date +%Y_%m_%d_%H:%M)

# How long is each run gonna last in simulated milliseconds
time_duration=100

if [ -n "$vtune" ]; then
    PREFIX="amplxe-cl -collect $vtune -data-limit=1000 -discard-raw-data -result-dir=$DEVITO_HOME/../profilings/vtune/$name-so$order-$vtune -resume-after=55 -search-dir=/tmp/devito-1000/"
elif [ -n "$advisor" ]; then
    PREFIX="advixe-cl -collect $advisor -data-limit=500 -project-dir=$DEVITO_HOME/../profilings/advisor/$name-so$order-$advisor -resume-after=55000 -search-dir=all:r=/tmp/devito-1000/ -run-pass-thru=--no-altstack"
elif [ -n "$cprofile" ]; then
    SUFFIX="-m cProfile -o profile.dat"
fi

if [[ "$mode" == "maxperf" || "$mode" == "dse" || "$mode" == "dle" ]]; then
    # Output directories
    export DEVITO_RESULTS=$DEVITO_OUTPUT/$system/$name-$arch-$timestamp
    mkdir -p $DEVITO_RESULTS
    # Record machine model
    cat /proc/cpuinfo | grep 'model name' | uniq > $DEVITO_RESULTS/core_model.txt
    # Run the benchmark
    $NUMACTL python $DEVITO_HOME/examples/seismic/benchmark/benchmark.py bench -bm $mode --repeats 1 -P $problem --tn $time_duration -a -d $grid $grid $grid -so $space_orders -to $time_orders -r $DEVITO_RESULTS
elif [ "$mode" == "srun" ]; then
    # Single shot run
    $PREFIX $NUMACTL python $SUFFIX $DEVITO_HOME/examples/seismic/benchmark/benchmark.py run -P $problem --tn $time_duration $dse $dle $at -d $grid $grid $grid -so $order -to 2
else
    echo "Unrecognised mode"
    exit
fi

if [ -n "$cprofile" ]; then
    gprof2dot -f pstats profile.dat | dot -Tpdf -o profile.pdf
    rm profile.dat
fi
