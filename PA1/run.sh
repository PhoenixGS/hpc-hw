#!/bin/bash

# run on 1 machine * 28 process, feel free to change it!
if (($2 <= 1000)); then
    srun -N 1 -n 1 --cpu-bind=none,verbose ./test.sh $*
else
    if (($2 <= 10000)); then
        srun -N 1 -n 28 --cpu-bind=none,verbose ./test.sh $*
    else
        srun -N 2 -n 56 --cpu-bind=none,verbose ./test.sh $*
    fi
fi