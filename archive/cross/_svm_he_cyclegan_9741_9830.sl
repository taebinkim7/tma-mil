#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=14-0:00:00
#SBATCH --partition=interactive
#SBATCH --output=slurm_logs/output-%x-%j.log
#SBATCH --mail-type=end
#SBATCH --mail-user=taebinkim@unc.edu

source ~/miniconda3/etc/profile.d/conda.sh

conda activate tma

python /datastore/lbcfs/labs/smarronlab/tkim/projects/tma-mil/run_cross_dataset.py \
--train-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741/mil/he \
--test-dir /datastore/lbcfs/labs/smarronlab/tkim/data/cyclegan/he/9830_to_9741/mil \
-m vgg16 -l block4_pool --cat er --cv-folds 5 \
--train-instance-size 800 --train-instance-stride 200 \
--test-instance-size 800 --test-instance-stride 400 \
--mi quantile -q 16 \
--save-train --load-train --n-jobs 16