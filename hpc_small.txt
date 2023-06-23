#!/bin/bash
#PBS -N TGV_KEPS_Check
#PBS -q small
#PBS -l select=1:ncpus=8
#PBS -j oe
#PBS -V
#PBS -o log_small.out

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE > ./pbsnodes_small
PROCS1=$(cat $PBS_NODEFILE | wc -l)

python code/taylor_green.py --ext-forcing --openmp --scheme=k_eps --eos=tait --pst-freq=50 --integrator=rk2 --integrator-dt-mul-fac=1 --re=100000 --nx=200 --tf=4.0 --c0-fac=20 --pfreq=168 -d ext_force_keps_tait_rk2_nx_200_re_100000_pst_50_output