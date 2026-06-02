#!/bin/bash

set -u

WORK_ROOT=/home/${USER}/guess
RESULT_DIR=${WORK_ROOT}/results/mpi_static_np

cd ${WORK_ROOT} || exit 1
mkdir -p ${RESULT_DIR}

echo "===== Compile mpi_static ====="
mpic++ -O2 -std=c++17 \
  correctness_guess_mpi_static.cpp train.cpp guessing.cpp md5.cpp \
  -o mpi_static

if [ $? -ne 0 ]; then
    echo "Compile failed."
    exit 1
fi

echo "===== Check executable ====="
ls -lh mpi_static

echo "===== Check dataset ====="
ls -lh /guessdata/Rockyou-singleLined-full.txt
if [ $? -ne 0 ]; then
    echo "Dataset missing: /guessdata/Rockyou-singleLined-full.txt"
    exit 1
fi

for NP in 1 2 4 8
do
    echo ""
    echo "========================================"
    echo "Run MPI static cyclic with NP=${NP}"
    echo "========================================"

    JOB_SCRIPT=qsub_mpi_static_np${NP}.sh
    OUT_FILE=mpi_static_np${NP}.o
    ERR_FILE=mpi_static_np${NP}.e

    rm -f ${OUT_FILE} ${ERR_FILE}

    cat > ${JOB_SCRIPT} <<EOF_JOB
#!/bin/sh
#PBS -N mpi_np${NP}
#PBS -e ${ERR_FILE}
#PBS -o ${OUT_FILE}
#PBS -l nodes=1:ppn=${NP}

MASTER_DIR=/home/\${USER}/guess
WORK_DIR=/home/\${USER}/mpi_static_work_np${NP}
EXE=mpi_static
NP=${NP}

NODES=\$(cat \$PBS_NODEFILE | sort | uniq)

for node in \$NODES; do
    ssh \${node} "mkdir -p \${WORK_DIR}" 1>&2
    scp master_ubss1:\${MASTER_DIR}/\${EXE} \${node}:\${WORK_DIR}/\${EXE} 1>&2
done

cd \${WORK_DIR} || exit 1

/usr/local/bin/mpiexec -np \${NP} -machinefile \$PBS_NODEFILE \${WORK_DIR}/\${EXE}
EOF_JOB

    chmod +x ${JOB_SCRIPT}

    echo "Submit ${JOB_SCRIPT}"
    JOB_ID=$(qsub ${JOB_SCRIPT})
    echo "Job ID: ${JOB_ID}"

    while true
    do
        STATE=$(qstat ${JOB_ID} 2>/dev/null | awk 'NR==3 {print $5}')
        if [ "${STATE}" = "C" ]; then
            echo "Job ${JOB_ID} completed."
            break
        fi
        if [ -z "${STATE}" ]; then
            echo "Job ${JOB_ID} disappeared from qstat; assume completed."
            break
        fi
        echo "Job ${JOB_ID} state=${STATE}, waiting..."
        sleep 10
    done

    sleep 2

    echo "===== stdout NP=${NP} ====="
    cat ${OUT_FILE}

    echo "===== stderr NP=${NP} ====="
    cat ${ERR_FILE}

    cp ${OUT_FILE} ${RESULT_DIR}/mpi_static_np${NP}.o
    cp ${ERR_FILE} ${RESULT_DIR}/mpi_static_np${NP}.e

    echo "Saved to ${RESULT_DIR}/mpi_static_np${NP}.o and .e"
done

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
grep -E "MPI size|Total guesses|Total cracked|Max guess time|Max hash time|Max generate time|Total wall time" ${RESULT_DIR}/mpi_static_np*.o

echo ""
echo "All MPI static cyclic runs finished."
