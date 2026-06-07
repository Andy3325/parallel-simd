#!/bin/sh
#PBS -N mpi_final
#PBS -o mpi_final.o
#PBS -e mpi_final.e
#PBS -l nodes=1:ppn=8
#PBS -V

MASTER_DIR=/home/${USER}/guess
WORK_DIR=/home/${USER}/mpi_final_work
EXE=main
NP=${NP:-8}
MODE=${MODE:-basic}
BATCH_SIZE=${BATCH_SIZE:-1}

NODES=$(cat $PBS_NODEFILE | sort | uniq)

for node in $NODES; do
    ssh ${node} "mkdir -p ${WORK_DIR} && rm -f ${WORK_DIR}/${EXE}" 1>&2
    scp master_ubss1:${MASTER_DIR}/${EXE} ${node}:${WORK_DIR}/${EXE} 1>&2
    ssh ${node} "chmod +x ${WORK_DIR}/${EXE}" 1>&2
done

cd ${WORK_DIR} || exit 1

/usr/local/bin/mpiexec -np ${NP} -machinefile $PBS_NODEFILE \
  ${WORK_DIR}/${EXE} --mode ${MODE} --batch-size ${BATCH_SIZE}
