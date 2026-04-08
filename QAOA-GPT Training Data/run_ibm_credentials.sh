#!/bin/bash -l
#SBATCH --job-name=ibm_credential
# speficity number of nodes 
#SBATCH -N 1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 1

# specify the walltime e.g 20 mins
#SBATCH -t 00:02:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruairi.hogan@ucdconnect.ie

# run from current directory
cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

module load python/3.7.4

python ibm_credentials.py
