#!/bin/bash

source /opt/intel/oneapi/setvars.sh

ulimit -s unlimited
echo "the start time is:"   $(date)  >> timing.log
DATE1=$(date +%s)

mpirun -np 32 ~/DeltaSpin-main/bin/vasp_ncl > tmp_log 2>&1

DATE2=$(date +%s)
echo "the end time is:"   $(date)   >> timing.log

diff=$((DATE2-DATE1))
printf "TIME COST: %d DAYS %02d:%02d:%02d" \
$((diff/86400)) $(((diff/3600)%24)) $(((diff/60)%60)) $(($diff %60)) >> timing.log
echo -e "\n\n" >> timing.log

sh ~/DeltaSpin-main/scripts/energy_force.sh >> tmp_log

