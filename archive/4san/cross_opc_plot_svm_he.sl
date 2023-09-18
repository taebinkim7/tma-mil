#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=interactive
#SBATCH --output=slurm_logs/run-%x-%j.log
#SBATCH --mail-type=fail
#SBATCH --mail-user=taebinkim@unc.edu

source ~/miniconda3/etc/profile.d/conda.sh

conda activate tma

python /datastore/lbcfs/labs/smarronlab/tkim/projects/tma-mil/run_cross_opc_plot.py \
--output-dir /datastore/lbcfs/labs/smarronlab/tkim/projects/tma-mil/san/plots \
--train-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741_san_9741_svd_gaussian_one_zero/mil/he \
--test-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741_san_cbcs_svd_gaussian_one_zero/mil/he \
--instance-size 400 \
--instance-stride 400 \
--cat er \
-c svm \
--mi quantile \
-q 16