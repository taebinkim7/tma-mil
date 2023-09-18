#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=interactive
#SBATCH --output=slurm_logs/output-%x-%j.log
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=taebinkim@unc.edu

source ~/miniconda3/etc/profile.d/conda.sh

git pull

conda activate tma

mkdir -p /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"/mil/he
cp /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"/mil/he/sample_images.csv \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"/mil/he/labels.csv \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"/mil/he
cp /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"/gen_csv.py \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"

mkdir -p /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"/mil/he
cp /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"/mil/he/sample_images.csv \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"/mil/he/labels.csv \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"/mil/he
cp /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"/gen_csv.py \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"

python /datastore/lbcfs/labs/smarronlab/tkim/projects/stain-mcmc/scripts/san_images.py \
--train-input-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"/images/he \
--train-output-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"/images/he \
--test-input-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"/images/he \
--test-output-dir /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"/images/he \
--extractor-type svd \
--n-jobs 50
