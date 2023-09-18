#! /bin/bash
CASE=gaussian_half_zero_uniform_train_mixup

# adapt images
jid1=$(sbatch --export TRAIN_DIR=9741,TEST_DIR=cbcs,CASE=$CASE san_he.sl)
jid2=$(sbatch --export TRAIN_DIR=cbcs,TEST_DIR=9741,CASE=$CASE san_he.sl)

# feature extraction
jid3=$(sbatch --export CASE=$CASE --dependency=afterok:${jid1: -6},${jid2: -6} vgg16_he.sl)

# MIL
sbatch --export CASE=$CASE --dependency=afterok:${jid3: -6} svm_he_9741_cbcs.sl
sbatch --export CASE=$CASE --dependency=afterok:${jid3: -6} svm_he_cbcs_9741.sl