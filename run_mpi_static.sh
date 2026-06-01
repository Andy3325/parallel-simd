#!/bin/sh
#PBS -N mpi_static
#PBS -e mpi_static.e
#PBS -o mpi_static.o
#PBS -l nodes=1:ppn=8

MASTER_DIR=/home/${USER}/guess
WORK_DIR=/home/${USER}/mpi_static_work
EXE=mpi_static
NP=8

NODES=$(cat $PBS_NODEFILE | sort | uniq)

for node in $NODES; do
    ssh ${node} "mkdir -p ${WORK_DIR}" 1>&2
    scp master_ubss1:${MASTER_DIR}/${EXE} ${node}:${WORK_DIR}/${EXE} 1>&2
done

cd ${WORK_DIR} || exit 1

/usr/local/bin/mpiexec -np ${NP} -machinefile $PBS_NODEFILE ${WORK_DIR}/${EXE}
