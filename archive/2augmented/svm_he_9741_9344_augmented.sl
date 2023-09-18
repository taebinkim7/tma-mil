#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=14-0:00:00
#SBATCH --partition=interactive
#SBATCH --output=slurm_logs/output-%x-%j.log
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=taebinkim@unc.edu

source ~/miniconda3/etc/profile.d/conda.sh

conda activate tma

python /datastore/lbcfs/labs/smarronlab/tkim/projects/tma-mil/run_cross_dataset.py \
--train-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741_augmented_svd/mil/he \
--test-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_9344_augmented_svd/mil/he \
-m vgg16 -l block4_pool --cat er --cv-folds 5 \
--train-instance-size 400 --train-instance-stride 400 \
--test-instance-size 400 --test-instance-stride 400 \
-q 16 \
--mi quantile \
--save-train --load-train --n-jobs 16
