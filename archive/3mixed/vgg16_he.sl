#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=2-0:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/run-%x-%j.log
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=taebinkim@unc.edu

source ~/miniconda3/etc/profile.d/conda.sh

conda activate tma

python /datastore/lbcfs/labs/smarronlab/tkim/projects/tma-mil/run_cnn_features.py \
-i /datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741_mixed_cbcs_svd/images/he \
-o /datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741_mixed_cbcs_svd/mil/he \
-m vgg16 -l block4_pool --instance-size 400 --instance-stride 400

python /datastore/lbcfs/labs/smarronlab/tkim/projects/tma-mil/run_cnn_features.py \
-i /datastore/lbcfs/labs/smarronlab/tkim/data/tma_cbcs_mixed_9741_svd/images/he \
-o /datastore/lbcfs/labs/smarronlab/tkim/data/tma_cbcs_mixed_9741_svd/mil/he \
-m vgg16 -l block4_pool --instance-size 400 --instance-stride 400