import json

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz
import ast
import json
import os
import random
import jsonpickle
import glob
from nearpy import Engine
from nearpy.filters import NearestFilter
from nearpy.hashes import RandomDiscretizedProjections

# constants
reverse_metrics=['ks', 't']


def default(o):
    if isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.float64):
        return float(o)
    raise TypeError

# functions
def write_csv(results, thresholds, metric_name, args):
    split_results = np.split(np.array(results), len(thresholds))
    keys = results[0].keys()
    if not os.path.exists('results/results.csv'):
        s = 'dataset,algorithm,metric,threshold' + ',' + ','.join(keys) + '\n'
    else:
        s = ''
    n=0
    for res in split_results:
        cols = [args.feature_file[0].split("/")[-1], args.algorithm, metric_name, thresholds[n]]
        for k in keys:
            cols.append(np.mean([x[k] for x in res]))
        s += ','.join([str(x) for x in cols]) + '\n'
        n += 1

    with open('results/'+args.saveas.split('/')[1]+'.txt', 'w') as res:
        d = {'results' : results, 'thresholds': thresholds, 'metric_name' : metric_name}
        s=json.dumps(d, default=default)
        res.write(s)


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def print_stats(data, metric):
    data = data.replace('inf', np.inf)
    data = data.replace('nan', np.nan)
    data = data[pd.isnull(data[metric]) == False]
    print(data[metric][0])
    infs=data[np.isfinite(data[metric]) == False]
    nans=data[np.isnan(data[metric])]
    print("Infinite: {0}".format(len(infs.index)))
    print("Nans: {0}".format(len(nans.index)))


def show_coefficients(classifier, clf, feature_names, top_features=20):
    if classifier == "svml":
        coef = clf.coef_.ravel()
    elif classifier == "rf":
        if isinstance(clf, Pipeline):
            clf = clf.named_steps['clf']
        coef = clf.feature_importances_
    elif classifier == "dt":
        export_graphviz(clf, out_file='tree.dot', feature_names=feature_names)
        coef = clf.feature_importances_
    elif classifier == 'xgb':
        coef = clf.feature_importances_
    else:
        return
    top_positive_coefficients = np.argsort(coef)[-top_features:][::-1]

    feature_names = np.array(feature_names)
    print(list(zip(feature_names[top_positive_coefficients], map(lambda x: x, sorted(coef, reverse=True)))))


def filter_by_metrics(features, metric_label, metric_value, labels, threshold):
    # filter by label and drop na

    labels = labels[labels[metric_value].isnull() == False].fillna(0).replace(np.inf, 99999999).replace(-np.inf, 99999999)

    if threshold is None:
        # use default label
        labels[metric_label] = labels[metric_label].apply(lambda x: 1 if str(x).strip() in [True, 'True'] else 0)
    else:
        if metric_label in reverse_metrics:
            labels[metric_label] = labels[metric_value].apply(
                lambda x: 1 if float(str(x).strip()) > float(threshold) else 0)
        else:
            labels[metric_label] = labels[metric_value].apply(
                lambda x: 1 if float(str(x).strip()) < float(threshold) else 0)

    feature_indices = list(features.index.tolist())
    labels_indices = list(labels.index.tolist())
    common_indices = [x for x in feature_indices if x in labels_indices]
    print("common " + str(len(common_indices)))
    # merge
    table = features.join(labels, sort=False)
    table = table.loc[common_indices]
    table = table.fillna(0).replace(np.inf, 99999999).replace(-np.inf, 99999999)
    return table


def plot_results(metric_label, exp_results, metric_thresholds, xlabels, saveas, args):
    font = {'family': 'normal',
            'size': 20}
    measures = {"F1": "F1", "AUC": "AUC"}
    matplotlib.rc('font', **font)
    ax = plt.gca()

    from matplotlib.ticker import FormatStrFormatter
    if metric_label in xlabels and xlabels[metric_label] == 'MCMC Iterations':
        pass
    elif metric_label == 't' or metric_label == 'ks':
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    min_value = 0.45
    for measure in measures.keys():
        mean=aggregate(np.mean, exp_results, len(metric_thresholds), measure)
        std=aggregate(np.std, exp_results, len(metric_thresholds), measure)
        plt.errorbar(metric_thresholds, mean, label=measures[measure],
                     marker='s', linewidth=3.0, yerr=std)
        min_value=min(min_value, np.min(mean))


    plt.xticks(metric_thresholds)
    plt.grid(True)
    plt.ylim((min_value, 1.05))
    plt.yticks(np.arange(np.round(min_value+0.1, 1), 1.01, 0.1))
    if metric_label in xlabels:
        plt.xlabel(xlabels[metric_label])
    plt.ylabel("Scores")
    # plt.legend()
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.11), ncol=4, prop={'size': 16})
    plt.tight_layout()
    # annotate(threshold, [x["Precision"] for x in results])
    if args is not None:
         write_csv(exp_results, metric_thresholds, metric_label, args)

    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()


def aggregate(f, arr, groups, field):
    if f == np.std:
        return [[np.mean([x[field] for x in part]) - np.min([x[field] for x in part])
                for part in np.split(np.array(arr), groups)],
                [np.max([x[field] for x in part]) - np.mean([x[field] for x in part])
                 for part in np.split(np.array(arr), groups)]]

    else:
        return [f([x[field] for x in part]) for part in np.split(np.array(arr), groups)]


def my_read_csv(file, index):
    f = open(file).readlines()
    df = pd.DataFrame([x.strip().split(',') for x in f[1:]], columns=f[0].strip().split(','))
    df = df.set_index(index)
    return df


def getRunConfig(model, metric):
    try:
        f=json.load(open("rf_params.json"))
        if type(model) == list:
            model=model[0]

        model=model.split('/')[-1].split('.')[0].replace("_features", "")
        return f[model][metric]
    except Exception as e:
        return dict()


def read_all_csvs(csv_files, prog_map, index):
    """
    :type csv_files: list
    :type prog_map: dict
    :type index: str
    """
    if csv_files is None:
        print("No files")
        exit(-1)
    full_data = None
    if not type(csv_files) == list:
        csv_files = [csv_files]
    for file in csv_files:
        if full_data is None:
            full_data = pd.read_csv(file, index_col=index).astype(np.float32)
        else:
            df = pd.read_csv(file, index_col=index).astype(np.float32)
            full_data=full_data.append(df)
    for i in full_data.index.tolist():
        try:
            id=i.split('/')[-2]
            if id not in prog_map:
                prog_map[id]= i.split('/')[-1].replace(".stan", "").replace(".template", "")
        except:
            pass
    return full_data


def update_features(original_features, files, ignored_indices, prog_map):
    # update table with additional features
    newdata = read_all_csvs(files, prog_map, index='program')
    for ig in ignored_indices:
        newdata = newdata.filter(set(newdata).difference(set(newdata.filter(regex=(ig)))))
    for i in list(newdata):
        if newdata[i].dtype == np.float64:
            newdata[i] = newdata[i].astype(np.float32)
    original_features = pd.merge(left=original_features, right=newdata, how='left', left_index=True, right_index=True)
    for i in list(original_features):
        if original_features[i].dtype == np.float64:
            original_features[i] = original_features[i].astype(np.float32)
    # original_features = original_features.join(newdata)
    original_features = original_features.replace('inf', np.inf)
    original_features = original_features.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)
    return original_features


def checkna(table, col=None):
    cols=list(table)
    #print(cols)
    for i in table.index.tolist():
        #print(i)
        if col is None:
            for c in cols:
                try:
                    if np.isreal(table.loc[i][c]) and not np.isfinite(table.loc[i][c]):
                        print('Col {0}'.format(c))
                        print(table.loc[i])
                        exit(1)
                except:
                    print(i)
                    print(table.loc[i])
                    exit(1)
        else:
            try:
                np.float32(str(table.loc[i][col]).strip())
            except:
                print(table.loc[i])
                exit(1)

def annotate(X, Y):
    ax=plt.gca()
    for i,j in zip(X,Y):
        ax.text(i,j,  "{:.2f}".format(j))


def reduce_pca(features):
    from sklearn.decomposition import PCA
    print("Original features: ", len(list(features)))
    pca = PCA(n_components=700)
    features_new=pca.fit_transform(features)
    features_new=pd.DataFrame(features_new, index=features.index)
    print("Reduced features: ", len(list(features_new)))
    return features_new


def balance(full_table, metric_label):
    # under sampling
    pos = full_table[full_table[metric_label] == 1].index
    neg = full_table[full_table[metric_label] == 0].index
    pos_samples = len(list(pos))
    neg_samples = len(list(neg))
    if pos_samples == 0 or neg_samples == 0:
        indices = list(full_table.index.tolist())
        print("Skipping balance, skewed data...")
    elif pos_samples < neg_samples:
        new_neg_samples = full_table[full_table[metric_label] == 0].sample(pos_samples).index
        indices = list(pos) + list(new_neg_samples)
    else:
        new_pos_samples = full_table[full_table[metric_label] == 1].sample(neg_samples).index
        indices = list(neg) + list(new_pos_samples)

    print("Balanced indices size: {0}".format(len(indices)))
    return indices


def test_train_split_class(full_table, indices, features, metric_label, prog_map, classname, classfile, train_size=1.0):
    # select programs and its mutants
    print(classname)
    progs=classfile[classname]
    print(progs)
    train_ind = list(filter(lambda x: prog_map[x] not in progs, indices))
    test_ind = list(filter(lambda x: prog_map[x] in progs, indices))
    
    if train_size < 1.0:
        new_train_size=int(len(train_ind)*train_size)
        print("Running with reduced training size. Original : {0}, Reduced : {1}".format(len(train_ind), new_train_size))
        train_ind=np.random.choice(train_ind, new_train_size)
    print(len(list(train_ind)))
    print(len(list(test_ind)))
    print(len(features))
    X_train =  features.loc[train_ind] #[list(features.loc[i]) for i in train_ind]
    Y_train = [full_table.loc[i][metric_label] for i in train_ind]
    X_test = features.loc[test_ind] #[list(features.loc[i]) for i in test_ind]
    Y_test = [full_table.loc[i][metric_label] for i in test_ind]
    print(len(X_train))
    print(len(X_test))
    return X_train, X_test, Y_train, Y_test, train_ind, test_ind


def test_train_validation_splits_by_template(indices, folds, prog_map, feature_file):
    assert folds > 2
    templates = list(set(prog_map.values()))
    print(templates)
    print(feature_file)
    print("Templates : {0}".format(len(templates)))
    split_points = np.linspace(0, len(templates), folds+1)
    cv_indices = []
    trains=[]
    tests=[]
    
    for f in random.sample(templates, folds):
        #cur_templates = templates[int(split_points[f]):int(split_points[f+1])]
        train_ind = []
        test_ind = []
        # train_ind = filter(lambda x: prog_map[x] not in f, indices)
        # test_ind = filter(lambda x: prog_map[x] in f, indices)
        for i, ind in enumerate(indices):
            if prog_map[ind] != f:
                train_ind.append(i)
            # if ind.split('_')[0] not in f:
            #     train_ind.append(i)

        for i, ind in enumerate(indices):
            # if ind.split('_')[0] in f:
            #     test_ind.append(i)
            if prog_map[ind] == f:
                test_ind.append(i)
        print(len(train_ind))
        print(len(test_ind))
        if len(test_ind) == 0:
            continue
        trains.append(pd.Index(train_ind))
        tests.append(pd.Index(test_ind))
        cv_indices.append((train_ind, test_ind))

    return cv_indices

def createLSH(dimensions, projections):
    nearest = NearestFilter(1)
    bin_width = 10
    projections = projections
    rbp = RandomDiscretizedProjections('rbp', projections, bin_width)
    rbp2 = RandomDiscretizedProjections('rbp2', projections, bin_width)
    rbp3 = RandomDiscretizedProjections('rbp3', projections, bin_width)
    rbp4 = RandomDiscretizedProjections('rbp4', projections, bin_width)

    engine = Engine(dimensions, lshashes=[rbp, rbp2, rbp3, rbp4], vector_filters=[nearest])
    return engine

def getDifferentPrograms(features, projections):
    engine=createLSH(len(list(features)), projections)
    indices_new=[]
    indices=list(features.index)
    np.random.shuffle(indices)
    for index in indices:
        rr=np.array(features.loc[index], dtype=np.float32)
        N = engine.neighbours(rr)
        if len(N) < 1:
            engine.store_vector(rr, data=index)
            indices_new.append(index)
    return indices_new
         
def test_train_split_template(full_table, indices, features, metric_label, ratio, shuffle_templates, prog_map:dict, train_size=1.0, projection_size=None):
    templates = list(set([x.split('_')[0] for x in indices]))
    if shuffle_templates:
        np.random.shuffle(templates)
    print("Templates : {0}".format(len(templates)))
    if type(ratio) == str:
        # filter out only given template name
        print(ratio)
        template_ids = list(filter(lambda x: prog_map[x] == ratio, indices))
        print('ids found:  ', len(template_ids))
        train_set_templates=list(set(indices) - set(template_ids))
        test_set_templates=template_ids

        if train_size < 1.0:
            new_train_size=int(len(train_set_templates)*train_size)
            print("Running with reduced training size. Original : {0}, Reduced : {1}".format(len(train_set_templates), new_train_size))
            train_set_templates=np.random.choice(train_set_templates, new_train_size)

        if projection_size is not None:
            print("Using projection size: " , projection_size)
            print("Old train indices: ", len(train_set_templates))
            print("Old test indices: ", len(test_set_templates))
            filtered_templates=getDifferentPrograms(features.loc[train_set_templates + test_set_templates], int(projection_size))
            train_set_templates=list(set(train_set_templates).intersection(set(filtered_templates)))
            test_set_templates=list(set(test_set_templates).intersection(set(filtered_templates)))
            print("New train indices: ", len(train_set_templates))
            print("New test indices: ", len(test_set_templates))
        
    else:
        if type(ratio) == float:
            split_point = int(ratio*len(templates))
        else:
            # leave x out
            split_point = len(templates) - ratio

        train_set_templates = templates[:split_point]
        test_set_templates = templates[split_point:]


    train_ind = train_set_templates
    test_ind = test_set_templates

    if prog_map is not None:
        print("Templates removed : " ,set([prog_map[x] for x in test_ind]))
    print("Train size: {0}, Test size: {1}".format(len(train_ind), len(test_ind)))

    X_train = features.loc[train_ind]
    Y_train = [full_table.loc[i][metric_label] for i in train_ind]
    X_test = features.loc[test_ind]
    Y_test = [full_table.loc[i][metric_label] for i in test_ind]
    #print("new columns:", len(list(X_train)))
    return X_train, X_test, Y_train, Y_test, train_ind, test_ind
