#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2-0:00:00
#SBATCH --partition=gpu
#SBATCH --output=run-%j.log
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=taebinkim@unc.edu

source ~/miniconda3/etc/profile.d/conda.sh

conda activate tma

python run_cnn_features.py \
-i /datastore/lbcfs/labs/smarronlab/tkim/data/cross/he/9830_to_9741 \
-o  /datastore/lbcfs/labs/smarronlab/tkim/data/cross/he/9830_to_9741/mil \
-m vgg16 -l block4_pool --instance-size 800 --instance-stride 400
