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

python /datastore/lbcfs/labs/smarronlab/tkim/projects/ImageMIL/run_cross_dataset.py \
--train-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741/mil/er \
--test-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_9830/mil/er \
-m vgg16 -l block4_pool --cat er --cv-folds 5 \
--instance-size 800 --instance-stride 400 --mi quantile_opc \
--save-train --load-train --n-jobs 16