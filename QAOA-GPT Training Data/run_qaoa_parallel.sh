#!/bin/bash -l
#SBATCH --job-name=QAOA_GEN
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=naoise.golden@ucdconnect.ie
#SBATCH --output=logs/qaoa_%j.out
#SBATCH --error=logs/qaoa_%j.err

module purge
module load python

# Go to your project directory
cd "$PROTOTYPE_DIR"

# Activate your virtual environment
source ./venv/bin/activate

# Make output folders
mkdir -p logs outputs

# Prevent each Python process from trying to use all CPU threads itself
# This is important when launching many processes in parallel
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Job started on $(hostname)"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "Start time = $(date)"

# Run IBM credentials setup once
python ibm_credentials.py

# Number of simultaneous runs to launch
N_RUNS=8

# Launch N_RUNS copies simultaneously
for i in $(seq 1 $N_RUNS); do
    OUTFILE="outputs/qaoa_dataset_${SLURM_JOB_ID}_${i}.jsonl"

    echo "Launching run $i -> $OUTFILE"

    srun --exclusive -N1 -n1 \
        python data_generation_hpc.py --output "$OUTFILE" \
        > "logs/run_${SLURM_JOB_ID}_${i}.out" \
        2> "logs/run_${SLURM_JOB_ID}_${i}.err" &
done

# Wait for all background jobs to finish
wait

echo "All runs finished at $(date)"
echo "Generated files:"
ls -lh outputs/qaoa_dataset_${SLURM_JOB_ID}_*.jsonl