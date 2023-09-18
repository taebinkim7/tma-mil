#! /bin/bash

# adapt images
jid1=$(sbatch mix_he_9741_cbcs.sl)
jid2=$(sbatch mix_he_cbcs_9741.sl)

# feature extraction
jid3=$(sbatch --dependency=afterok:${jid1: -6},${jid2: -6} vgg16_he.sl)

# MIL
sbatch --dependency=afterok:${jid3: -6} svm_he_9741_cbcs.sl
sbatch --dependency=afterok:${jid3: -6} svm_he_cbcs_9741.sl