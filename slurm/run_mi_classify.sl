#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=14-0:00:00
#SBATCH --partition=allnodes
#SBATCH --output=run-%j.log
#SBATCH --mail-type=end
#SBATCH --mail-user=taebinkim@unc.edu

source ~/miniconda3/etc/profile.d/conda.sh

conda activate mil

python run_mi_classify.py -o /datastore/lbcfs/labs/smarronlab/tkim/data/tma_cbcs/mil -m vgg16 -l block4_pool \
--cat er --cv-folds 5 --instance-size 800 --instance-stride 400 --mi quantile --n-jobs 8
