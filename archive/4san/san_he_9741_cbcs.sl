#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=allnodes
#SBATCH --output=slurm_logs/output-%x-%j.log
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=taebinkim@unc.edu

source ~/miniconda3/etc/profile.d/conda.sh

git pull

conda activate tma

train_dir=9741
test_dir=cbcs

mkdir -p /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$train_dir"_san_"$train_dir"_svd/mil/he
cp /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$train_dir"/mil/he/sample_images.csv \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$train_dir"/mil/he/labels.csv \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$train_dir"_san_"$train_dir"_svd/mil/he
cp /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$train_dir"/gen_csv.py \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$train_dir"_san_"$train_dir"_svd

mkdir -p /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$test_dir"_san_"$train_dir"_svd/mil/he
cp /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$test_dir"/mil/he/sample_images.csv \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$test_dir"/mil/he/labels.csv \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$test_dir"_san_"$train_dir"_svd/mil/he
cp /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$test_dir"/gen_csv.py \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$test_dir"_san_"$train_dir"_svd

python /datastore/lbcfs/labs/smarronlab/tkim/projects/stain-mcmc/scripts/san_images.py \
--train-input-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$train_dir"/images/he \
--train-output-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$train_dir"_san_"$train_dir"_svd/images/he \
--test-input-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$test_dir"/images/he \
--test-output-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$test_dir"_san_"$train_dir"_svd/images/he \
--extractor-type svd \
--n-jobs 50
