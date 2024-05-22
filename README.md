# tma-mil - Cross-dataset predictions of breast cancer phenotypes by image-based multiple instance learning

The methods implemented here include those discussed in the following submission:

T. Kim, B. C. Calhoun, Y. Li, A. Thennavan, M. A. Troester, L. A. Carey, W. F. Symmans, T. O. Nielsen, S. Leung, J. S. Marron, and C. M. Perou, “Cross-dataset predictions of breast cancer phenotypes by image-based multiple instance learning,” npj Breast Cancer, 2024.

Part of this repository has been adopted from https://github.com/hdcouture/ImageMIL.

## Setup

Basic installation requires a number of python packages, which are most easily installed with conda:

```
conda install -c conda-forge numpy scipy tensorflow scikit-learn scikit-image cudnn
```
Then install the python package `wdwd` from https://github.com/taebinkim7/weighted-dwd to implement Distance Weighted Discrimination (DWD).

## GPU Setup

If you encounter a problem when registering TensorFlow with GPU with the warning "Could not load dynamic library 'libcudnn.so.X'", try adding
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cudnn/library
```
in the .bashrc file.

## Data Setup

The above referenced publication used several dasets including the one from the [Carolina Breast Cancer Study](http://cbcs.web.unc.edu/for-researchers/).  You may apply for access to this data set.

Running this code requires two files: labels.csv and sample_images.csv.

labels.csv should use the following format for up to N samples and K classification tasks:
```
sample,class1,class2,...,classK
sample1,label11,label12,...,label1K
...
sampleN,labelN1,labelN2,...,labelNK
```

Each class can be binary or multi-class.  Any string or number can be used to identify the classes.

sample_images.csv allows one or more image files to be specified for each sample:
```
sample1,image11,image12,...,image1M
...
sampleN,imageN1,imageN2,...,imageNM
```

Each sample may have a different number of associated images.

If a specific train/test split is needed, a file or files may be provided in the following format:
```
sample1,train
sample2,train
sample3,test
...
sampleN,train
```

## Example Usage for CBCS

```
python setup_cbcs.py -i CBCS/images/ -o CBCS_out/ --spreadsheet CBCS.csv
python run_cnn_features.py -i CBCS/images/ -o CBCS_out/ -m vgg16 -l block4_pool --instance-size 800 --instance-stride 400
python run_mi_classify.py -o CBCS_out/ -m vgg16 -l block4_pool --cat grade1vs3 --cv-folds 5 --instance-size 800 --instance-stride 400 --mi quantile
python run_mi_classify.py -o CBCS_out/ -m vgg16 -l block4_pool --cat BasalvsNonBasal,er,ductal_lobular,ror-high --sample-weight grade12vs3 --group grade12vs3 --cv-folds 5 --instance-size 800 --instance-stride 400 --mi quantile
```
