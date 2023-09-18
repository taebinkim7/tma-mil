#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=14-0:00:00
#SBATCH --partition=interactive
#SBATCH --output=slurm_logs/run-%x-%j.log
#SBATCH --mail-type=fail
#SBATCH --mail-user=taebinkim@unc.edu

source ~/miniconda3/etc/profile.d/conda.sh

conda activate tma

python /datastore/lbcfs/labs/smarronlab/tkim/projects/tma-mil/run_km_plot.py \
-o /datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741/mil/he \
--cat surv_3yrs surv_5yrs surv_7yrs \
-c svm \
--random-state 111
