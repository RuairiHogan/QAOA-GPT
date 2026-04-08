#!/bin/bash -l
#SBATCH --job-name=qaoa_dataset
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-19%20


# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruairi.hogan@ucdconnect.ie

# run from current directory
cd $SLURM_SUBMIT_DIR

# command to use
hostname

set -euo pipefail

# -------------------------------------------------------------------
# Adjust these paths
# -------------------------------------------------------------------
SUBMIT_DIR="/home/people/21432816/data_generation"
SCRIPT_DIR="${SUBMIT_DIR}"
SCRIPT_NAME="data_generation_hpc.py"   # change to your actual python filename
VENV_PATH="${SUBMIT_DIR}/.venv"  # change to your virtualenv path
OUT_DIR="${SUBMIT_DIR}/outputs"
LOG_DIR="${SUBMIT_DIR}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# -------------------------------------------------------------------
# Unique index for this task
# -------------------------------------------------------------------
IDX="${SLURM_ARRAY_TASK_ID}"

# Unique output file for this task
OUTPUT_FILE="${OUT_DIR}/qaoa_dataset_part_${IDX}.jsonl"

echo "=== QAOA ARRAY JOB ==="
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Array Task ID:  ${IDX}"
echo "Node:           $(hostname)"
echo "Working dir:    ${SCRIPT_DIR}"
echo "Output file:    ${OUTPUT_FILE}"
echo "======================"

# -------------------------------------------------------------------
# Load Python module if your cluster uses modules
# Replace/remove this section as needed
# -------------------------------------------------------------------
module purge
module load python

# -------------------------------------------------------------------
# Activate environment
# -------------------------------------------------------------------
cd "${SCRIPT_DIR}"
source "${VENV_PATH}/bin/activate"

echo "Python: $(which python)"
python --version

# -------------------------------------------------------------------
# Run the script
# -------------------------------------------------------------------
python "${SCRIPT_NAME}" --output "${OUTPUT_FILE}"

echo "Finished task ${IDX}"

