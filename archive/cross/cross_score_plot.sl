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

TRAIN_DIR=9741
TEST_DIR=cbcs
CASE=uniform_train_mixup

python /datastore/lbcfs/labs/smarronlab/tkim/projects/tma-mil/run_cross_score_plot.py \
--output-dir /datastore/lbcfs/labs/smarronlab/tkim/projects/tma-mil/cross/plots \
--train-dirs /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_normalized_"$TRAIN_DIR"_svd/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_augmented_svd/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_mixed_"$TEST_DIR"_svd/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"/mil/he \
--test-dirs /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"_normalized_"$TRAIN_DIR"_svd/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"/mil/he \
--instance-size 400 \
--instance-stride 400 \
--cat er \
-c svm \
--mi quantile \
-q 16

TRAIN_DIR=cbcs
TEST_DIR=9741
CASE=uniform_train_mixup

python /datastore/lbcfs/labs/smarronlab/tkim/projects/tma-mil/run_cross_score_plot.py \
--output-dir /datastore/lbcfs/labs/smarronlab/tkim/projects/tma-mil/cross/plots \
--train-dirs /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_normalized_"$TRAIN_DIR"_svd/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_augmented_svd/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_mixed_"$TEST_DIR"_svd/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TRAIN_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"/mil/he \
--test-dirs /datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"_normalized_"$TRAIN_DIR"_svd/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"/mil/he \
/datastore/lbcfs/labs/smarronlab/tkim/data/tma_"$TEST_DIR"_san_"$TRAIN_DIR"_svd_"$CASE"/mil/he \
--instance-size 400 \
--instance-stride 400 \
--cat er \
-c svm \
--mi quantile \
-q 16