import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from joblib import load
from sklearn.decomposition import PCA
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival


parser = argparse.ArgumentParser(description='Generate KM plot.')
parser.add_argument('--in-dir', '-i', required=True, help='input directory')
parser.add_argument('--out-dir', '-o', required=True, help='output directory')
parser.add_argument('--mi',
                    help='MI type (none, median, max, quantile, quantile_mean, quantile_pca, quantile_mean_pca, quantile_opc); default: none (compute mean across images)')
parser.add_argument('--classifier', '-c', help='classifier (logistic, svm, or dwd); default: svm')
parser.add_argument('--cats', nargs= '+', help='label category used to train')
parser.add_argument('--instance-size', help='instance size')
parser.add_argument('--instance-stride', help='instance stride')
parser.add_argument('--quantiles', '-q', help='Number of quantiles; default: 16')
parser.add_argument('--level', '-l', help='level of classification (bag, instance)')
parser.add_argument('--random-state', help='random state for splitting datasets')
args = parser.parse_args()
in_dir = args.in_dir
if len(in_dir) > 1 and in_dir[-1] != '/':
    in_dir += '/'
out_dir = args.out_dir
if len(out_dir) > 1 and out_dir[-1] != '/':
    out_dir += '/'
mi_type = args.mi
classifier = args.classifier
cats = args.cats
instance_size = args.instance_size
instance_stride = args.instance_stride
quantiles = args.quantiles
random_state = args.random_state
level = args.level

if mi_type is None:
    mi_type = 'quantile'

if instance_size is None:
    instance_size = 400

if instance_stride is None:
    instance_stride = 400

if quantiles is None:
    quantiles = 16

def generate_km_plot(cat, prediction, surv_mos, surv_event):
    for surv_label in reversed(range(2)):
        mask = prediction == surv_label
        surv_time, surv_prob = kaplan_meier_estimator(surv_event[mask], surv_mos[mask])

        plt.step(surv_time, surv_prob, where="post",
                 c='tab:orange' if surv_label == 1 else 'tab:blue',
                 label=f"{cat} {'pos' if surv_label == 1 else 'neg'}")

    data_y = []
    for i in range(len(surv_mos)):
        data_y.append((surv_event[i], surv_mos[i]))
    data_y = np.array(data_y, dtype=[('Status', '?'), ('Survival months', '<f8')])
    chisq, pvalue, stats, covar = compare_survival(data_y, prediction, return_stats=True)
    plt.title(f'Log-rank test p-value: {pvalue: .4f}')
    plt.ylabel('Survival probability')
    plt.xlabel('Months')
    plt.legend(loc='best')


for cat in cats:
    data = load(in_dir + '_data_' + cat + '_random_state_' + random_state)
    model = load(in_dir + '_' + mi_type + '_' + classifier + '_' + cat + '_i' + str(instance_size) + '-'
                 + str(instance_stride) + '_q' + str(quantiles) + '_first')
    labels = pd.read_csv(in_dir + 'labels.csv', index_col=0)

    (idx_train, X_train, y_train), (idx_test, X_test, y_test) = data

    train_prediction = np.argmax(model.predict(X_train), axis=1)
    train_surv_mos = labels.loc[idx_train, 'surv_mos']
    train_surv_event = labels.loc[idx_train, 'surv_event'] == 1

    generate_km_plot(cat, train_prediction, train_surv_mos, train_surv_event)
    plt.savefig(out_dir + 'plots/train_km_plot_' + classifier + '_' + cat + '.png')
    plt.close()

    test_prediction = np.argmax(model.predict(X_test), axis=1)
    test_surv_mos = labels.loc[idx_test, 'surv_mos']
    test_surv_event = labels.loc[idx_test, 'surv_event'] == 1

    generate_km_plot(cat, test_prediction, test_surv_mos, test_surv_event)
    plt.savefig(out_dir + 'plots/test_km_plot_' + classifier + '_' + cat + '.png')
    plt.close()

