import os
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from joblib import dump, load
from sklearn.decomposition import PCA
from wdwd import WDWD

parser = argparse.ArgumentParser(description='Refit classifier.')
parser.add_argument('--out_dir', '-o', required=True, help='output directory')
parser.add_argument('--cat', help='label category used to train')
parser.add_argument('--classifier', '-c', help='classifier (logistic, svm, or dwd); default: svm')
parser.add_argument('--random-state', help='random state for splitting datasets')
args = parser.parse_args()
out_dir = args.out_dir
if len(out_dir) > 1 and out_dir[-1] != '/':
    out_dir += '/'
cat = args.cat
classifier = args.classifier
random_state = args.random_state


data = load(out_dir + '_data_' + cat + '_random_state_' + random_state)
# sil = load(out_dir + '_' + classifier + '_' + cat)
sil = load(out_dir + '_' + mi_type + '_' + classifier + '_' + cat_name + '_i' + str(instance_size) + '-' + str(instance_stride) + '_q' + str(quantiles))

(X_train, y_train), (X_test, y_test) = data


def flatten_bag_data(X, y):
    bags = [np.asmatrix(bag) for bag in X]
    X = np.vstack(bags)
    y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1))) for bag, cls in zip(bags, y)])

    return np.array(X), np.array(y).reshape(-1)


X_train, y_train = flatten_bag_data(X_train, y_train)
X_test, y_test = flatten_bag_data(X_test, y_test)

if classifier == 'svm':
    model = sklearn.svm.LinearSVC(C=sil.C, class_weight='balanced')
elif classifier == 'dwd':
    model = WDWD(C=sil.C)

model.fit(X_train, y_train)
model_path = out_dir + '_refitted_' + classifier + '_' + cat
dump(model, model_path)