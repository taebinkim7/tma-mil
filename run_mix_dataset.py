import os
import sys
import argparse
import numpy as np
import sklearn.model_selection
import sklearn.metrics
from joblib import dump, load

import util
from linear_classifier import LinearClassifier
from sil import SIL

import warnings
warnings.filterwarnings("ignore")


class ResultsReport:
    def __init__(self, label_names=None):
        self.res = {}
        self.label_names = label_names

    def add(self, metric, result):
        if metric not in self.res:
            self.res[metric] = []
        self.res[metric].append(result)

    def print_summary(self, metric=None):
        if metric is None:
            # for metric in sorted(self.res.keys()):
            for metric in self.res.keys():
                if metric != 'confusion':
                    self.print_summary(metric)
            self.print_summary('confusion')
            return
        if metric != 'confusion':
            mean = np.mean(self.res[metric])
            std = np.std(self.res[metric])
            ste = std / np.sqrt(len(self.res[metric]) - 1) if len(self.res[metric]) > 1 else 0.0
            print('%s %f %f %f' % (metric, mean, std, ste))
        else:
            print('confusion')
            print(('%s ' * len(self.label_names)) % tuple(self.label_names))
            print(sum(self.res['confusion']))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test MI classifier.')
    parser.add_argument('--data-dirs', nargs='+', required=True, help='data directories')
    parser.add_argument('--model', '-m', required=True, help='CNN model')
    parser.add_argument('--layer', '-l', required=True, help='CNN layer')
    parser.add_argument('--instance-size', help='instance size')
    parser.add_argument('--instance-stride', help='instance stride')
    parser.add_argument('--pool-size', '-p', help='mean pooling size')
    parser.add_argument('--cat', help='label categories to train (comma separated); default: all')
    parser.add_argument('--calibrate', action='store_true', help='calibrate classifier')
    parser.add_argument('--metric', help='metric to optimize during parameter search (accuracy, balanced_accuracy, roc_auc); default: accuracy')
    parser.add_argument('--classifier', '-c', help='classifier (logistic, svm, dwd); default: svm')
    parser.add_argument('--kernel', help='SVM kernel; default: linear')
    parser.add_argument('--mi', help='MI type (none, median, quantile, quantile_mean, quantile_pca, quantile_mean_pca, quantile_opc); default: none (compute mean across images)')
    parser.add_argument('--agg', help='Aggregate type (svm, dwd); default: svm')
    parser.add_argument('--quantiles', '-q', help='Number of quantiles; default: 16')
    parser.add_argument('--cv-fold-files', help='cross-validation fold files')
    parser.add_argument('--cv-folds', help='cross-validation folds')
    parser.add_argument('--cv-lno', help='cross-validation leave n out')
    parser.add_argument('--random-state', help='random state for splitting datasets')
    parser.add_argument('--save-train', action='store_true', help='save trained classifier')
    parser.add_argument('--load-train', action='store_true', help='load trained classifier')
    parser.add_argument('--n-jobs', help='number of parallel threads')
    parser.add_argument('--n-components', help='number of principal components')
    args = parser.parse_args()
    data_dirs = args.data_dirs
    for i, data_dir in enumerate(data_dirs):
        if len(data_dir) > 1 and data_dir[-1] != '/':
            data_dir += '/'
        data_dirs[i] = data_dir
    model_name = args.model
    layer = args.layer
    instance_size = args.instance_size
    instance_stride = args.instance_stride
    pool_size = args.pool_size
    categories = args.cat
    metric = args.metric
    calibrate = args.calibrate
    classifier = args.classifier
    kernel = args.kernel
    mi_type = args.mi
    agg_type = args.agg
    quantiles = args.quantiles
    cv_fold_files = args.cv_fold_files
    cv_folds = args.cv_folds
    cv_lno = args.cv_lno
    random_state = args.random_state
    save_train = args.save_train
    load_train = args.load_train
    n_jobs = args.n_jobs
    n_components = args.n_components

    if calibrate is None:
        calibrate = False
    else:
        calibrate = bool(calibrate)
        print(f'Calibrate: {calibrate}')

    if classifier is None:
        classifier = 'svm'

    if agg_type is None:
        agg_type = 'svm'

    if quantiles is None:
        quantiles = 16

    if save_train is None:
        save_train = False
    else:
        save_train = bool(save_train)

    if load_train is None:
        load_train = False
    else:
        load_train = bool(load_train)

    if n_jobs is not None:
        n_jobs = int(n_jobs)

    if random_state is not None:
        random_state = int(random_state)

    if n_components is not None:
        n_components = int(n_components)

    # load data
    feats, samples, cats, labels, label_names, groups = util.load_multi_data(data_dirs, model_name, layer, pool_size, instance_size, instance_stride, mi_type, categories, True)

    # set parameters
    options = {}
    if kernel is not None:
        options['kernel'] = kernel
    else:
        options['kernel'] = 'linear'
    if classifier is not None:
        options['classifier'] = classifier
    if mi_type is not None:
        options['predict_type'] = mi_type
    if agg_type is not None:
        options['agg_type'] = agg_type
    if metric is not None:
        options['metric'] = metric
    if n_components is not None:
        options['n_components'] = n_components

    # train classifier
    idx_all = np.arange(len(samples))
    idx_sliced = [np.random.RandomState(seed=random_state).permutation(idx_all[slice(*bidx)]) for bidx in util.slices(groups)]
    idx_train = np.concatenate([idx[:int(0.8 * len(idx))] for idx in idx_sliced])
    idx_test = np.concatenate([idx[int(0.8 * len(idx)):] for idx in idx_sliced])
    for c, cat_name in enumerate(cats):
        print(cat_name)
        res = ResultsReport(label_names[c])
        idx_train = [i for i in idx_train if (labels[i, c] != -1) & (samples[i] in feats)]
        idx_test = [i for i in idx_test if (labels[i, c] != -1) & (samples[i] in feats)]
        X_train = [feats[samples[i]] for i in idx_train]
        y_train = labels[idx_train, c]
        X_test = [feats[samples[i]] for i in idx_test]
        y_test = labels[idx_test, c]

        # load trained classifier
        model_path = data_dir + '_' + mi_type + '_' + classifier + '_' + cat_name + '_i' + str(instance_size) + '-' + str(instance_stride) + '_q' + str(quantiles)
        if load_train and os.path.exists(model_path):
            model = load(model_path)
        elif mi_type is None:
            model = LinearClassifier(n_jobs=n_jobs, **options)
            model.fit(X_train, y_train, calibrate=calibrate, param_search=True)
        elif mi_type in ['median', 'max']:
            model = SIL(n_jobs=n_jobs, **options)
            model.fit(X_train, y_train, calibrate=calibrate, param_search=True)
        elif 'quantile' in mi_type:
            if quantiles is not None:
                options['quantiles'] = int(quantiles)
            model = SIL(n_jobs=n_jobs, **options)
            model.fit(X_train, y_train, calibrate=calibrate, param_search=True)

        p_predict = model.predict(X_test)
        y_predict = np.argmax(p_predict, axis=1)
        acc = sklearn.metrics.accuracy_score(y_test, y_predict)
        if len(y_test) == 1:
            auc = 0.0
        elif len(np.unique(y_train)) == 2:
            auc = sklearn.metrics.roc_auc_score(y_test, p_predict[:, 1])
        else:
            auc = 0.0
            for i in range(p_predict.shape[1]):
                auc += sklearn.metrics.roc_auc_score(y_test == i, p_predict[:, i])
            auc /= p_predict.shape[1]
        kappa = sklearn.metrics.cohen_kappa_score(y_test, y_predict)
        classes = np.unique(y_train)
        np.sort(classes)
        confusion = sklearn.metrics.confusion_matrix(y_test, y_predict, labels=classes)
        res.add('acc', acc)
        # res.add('auc',auc)
        # res.add('kappa',kappa)
        if len(label_names[c]) == 2:
            res.add('sensitivity',
                    float(np.logical_and(y_test == 1, y_predict == y_test).sum()) / (y_test == 1).sum())
            res.add('specificity',
                    float(np.logical_and(y_test != 1, y_predict == y_test).sum()) / (y_test != 1).sum())
        res.add('auc', auc)
        res.add('kappa', kappa)
        res.add('confusion', confusion)

        print('accuracy %f auc %f' % (acc, auc))
        print(confusion)

        if save_train:
            model_path = data_dir + '_' + mi_type + '_' + classifier + '_' + cat_name + '_i' + str(instance_size) + '-' + str(instance_stride) + '_q' + str(quantiles)
            dump(model, model_path)

        print(f'Train instance size-stride: {instance_size}-{instance_stride}')
        print(f'Test instance size-stride: {instance_size}-{instance_stride}')
        print(f'Quantiles: {quantiles}')
        print('Cross-validation results')
        res.print_summary()
