#!/usr/bin/env python3
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import RFECV
from utils import *

import matplotlib
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import validation_curve, learning_curve, RandomizedSearchCV

from treeinterpreter import treeinterpreter as ti

import pickle
pd.set_option('expand_frame_repr', False)


def read_args():
    parser = argparse.ArgumentParser(description='ProbFuzz')
    parser.add_argument('-f', nargs='+', dest='feature_file')
    parser.add_argument('-fo', dest='feature_other', nargs='+')
    parser.add_argument('-l', nargs='+', dest='labels_file')
    parser.add_argument('-m', dest='metric')
    parser.add_argument('-th', dest='threshold', default=None, type=float)
    parser.add_argument('-ignore_vi', dest='ignore_vi', action='store_true')
    parser.add_argument('-split', dest='split_ratio', default=0.8, type=float)
    parser.add_argument('-tname', dest='split_template_name')
    parser.add_argument('-st', dest='split_by_template', action='store_true')
    parser.add_argument('-b', dest='balance', action='store_true', help='balance by subsampling')
    parser.add_argument('-bw', dest='balance_by_weight', action='store_true', help='balance by weight')
    parser.add_argument('-a', dest='algorithm', default='rf')
    parser.add_argument('-cv', dest='cv', action='store_true')
    parser.add_argument('-cv_temp', dest='cv_template', action='store_true', help='cross validation by template split')
    parser.add_argument('-plt', dest='plot', action='store_true')
    parser.add_argument('-validation', dest='validation', action='store_true')
    parser.add_argument('-learning', dest='learning', action='store_true')
    parser.add_argument('-grid', dest='grid', action='store_true')
    parser.add_argument('-suf', dest='metrics_suffix', default=None)
    parser.add_argument('-runtime', dest='runtime', action='store_true')
    parser.add_argument('-predict', nargs='+', dest='predict')
    parser.add_argument('--tree', dest='tree', action='store_true')
    parser.add_argument('--train_by_size', dest='train_by_size', action='store_true')
    parser.add_argument('-class', dest='split_class')
    parser.add_argument('-shuffle', dest='shuffle', action='store_true')
    parser.add_argument('-saveas', dest='saveas', default=None)
    parser.add_argument('-feature_select', dest='feature_select', action='store_true')
    parser.add_argument('-plt_temp', dest='plt_template', action='store_true')
    parser.add_argument('-warmup', dest='warmup', action='store_true')
    parser.add_argument('-stratify', dest='stratify_data', action='store_true')
    parser.add_argument('-special', dest='special_index')
    parser.add_argument('-ignore', dest='ignore', nargs='+')
    parser.add_argument('-keep', dest='keep', nargs='+')
    parser.add_argument('-selected', dest='selected')
    parser.add_argument('-with_noise', dest='with_noise', action='store_true')
    parser.add_argument('-tfpn', dest='tfpn', default=None)
    parser.add_argument('-allpreds', action='store_true')
    parser.add_argument('-testf', dest='test_features', nargs='+', help='external test features')
    parser.add_argument('-testl', dest='test_labels', help='external test labels')
    parser.add_argument('-rt_iter', dest='runtime_iteration')
    parser.add_argument('-train_size', dest='train_size', help='Run with reduced training size', default=1.0, type=float)
    parser.add_argument('-metric_filter', dest='metric_filter')
    parser.add_argument('-projection_size', dest='projection_size', default=None)
    parser.add_argument('-max_motifs', dest='max_motifs', default=0, type=int)
    parser.add_argument('-nooverlap', dest='nooverlap', type=float)

    args = parser.parse_args()
    print(args)
    return args


def split_dataset(X, Y, metric, ratio, shuffle, stratify_data, threshold):
    if stratify_data:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio,
                                                            shuffle=shuffle)
        print('Train (before) : ', sum([x<threshold for x in Y_train]), sum([x>threshold for x in Y_train]))
        print('Test (before) : ', sum([x<threshold for x in Y_test]), sum([x>threshold for x in Y_test]))

        X_train, Y_train = stratify(X_train, Y_train, metric)
        X_test, Y_test = stratify(X_test, Y_test, metric)

        print('Train (after) : ', sum([x < threshold for x in Y_train]), sum([x >= threshold for x in Y_train]))
        print('Test (after) : ', sum([x < threshold for x in Y_test]), sum([x >= threshold for x in Y_test]))
        if threshold is not None:
            Y_train = [float(y < threshold) for y in Y_train]
            Y_test = [float(y < threshold) for y in Y_test]
        return X_train, X_test, Y_train, Y_test
    else:
        return train_test_split(X, Y, train_size=ratio, shuffle=shuffle)


def stratify(X, Y, metric):
    bins = np.histogram(Y, range=(0.0, 2.0) if metric == 'wass' else (1.0, 2.0), bins=5)

    digs = np.digitize(Y, bins[1])
    bin_choices = [np.random.choice(list(set(digs))) for _ in range(len(Y))]
    indices = [np.random.choice(list(filter(lambda x: digs[x] == bin, range(len(Y))))) for bin in bin_choices]
    other_indices = list(set(range(len(Y))).difference(set(indices)))
    if isinstance(X, pd.DataFrame):
        Xnew = X.loc[indices+other_indices]
    else:
        Xnew = [X[i] for i in indices + other_indices]
    if isinstance(Y, pd.DataFrame):
        Ynew = Y.loc[indices + other_indices]
    else:
        Ynew = [Y[i] for i in indices + other_indices]


    return Xnew, Ynew

def feature_to_ast(feature_name):
    ast_nodes = []
    node_id_dict = {"33":"FunctionsContext", "61":"TransposeContext", "16":"BracketsContext", "56":"SubsetContext", "26":"ExprContext", "44":"NeContext", "32":"Function_declContext", "60":"TransformedparamContext", "40":"LoopcompContext", "36":"GtContext", "22":"DtypeContext", "66":"VectorContext", "67":"VectorDIMSContext", "51":"QueryContext", "29":"FparamsContext", "45":"NumberContext", "42":"MinusopContext", "34":"GeneratedquantitiesContext", "31":"Function_callContext", "54":"StatementContext", "39":"LimitsContext", "68":"__class__", "17":"DataContext", "20":"DistexprContext", "57":"TemplateContext", "43":"MulopContext", "52":"RefContext", "12":"ArrayContext", "49":"PrimitiveContext", "25":"ExponopContext", "18":"DeclContext", "47":"ParamContext", "15":"BlockContext", "21":"DivopContext", "13":"Array_accessContext", "27":"For_loopContext", "37":"If_stmtContext", "63":"ValContext", "41":"LtContext", "62":"UnaryContext", "59":"TransformeddataContext", "46":"ObserveContext", "30":"FunctionContext", "10":"AddopContext", "11":"AndContext", "19":"DimsContext", "58":"TernaryContext", "38":"LeqContext", "50":"PriorContext", "14":"AssignContext", "24":"EqContext", "64":"VecdivopContext", "28":"FparamContext", "48":"ParamsContext", "55":"StringContext", "65":"VecmulopContext", "53":"Return_or_param_typeContext", "23":"Else_blkContext", "35":"GeqContext", "76":"ParamsContext", "91":"MulopContext", "98":"StatementContext","3":"DimsContext", "81":"Array_accessContext", "9":"TransformeddataContext","92":"BracketsContext","0":"Context"}
    for feature_id in feature_name.split("_"):
        if feature_id in node_id_dict:
            ast_nodes.append(node_id_dict[feature_id][:-7])
        else:
            ast_nodes.append(feature_id)
    return "-".join(ast_nodes)


def predict(clf, X_test, Y_test, weighted=False, allpreds=False):
    stats = dict()
    stats["True"] = sum([x == 1 for x in Y_test])
    stats["False"] = sum([x == 0 for x in Y_test])
    print("True:" +str(stats["True"]))
    print("False:" + str(stats["False"]))
    stats["Diff"] = (stats["True"]+0.0)/len(Y_test)
    start=time.time()
    predictions = clf.predict(X_test)
    end=time.time()
    print('Total prediction time: ', (end-start))
    print('Predicting for', len(predictions))
    print('Prediction time per instance', ((end-start+0.0)/len(predictions)))
    # if isinstance(clf, xgb.XGBClassifier):
    #     predictions = [round(v) for v in predictions]

    prediction_probs = clf.predict_proba(X_test)

    error = sum([pp[0] != pp[1] for pp in zip(predictions, Y_test)])
    correct = sum([pp[0] == pp[1] for pp in zip(predictions, Y_test)])
    stats["Error"] = error / float(len(Y_test))
    stats["Accuracy"] = correct / float(len(Y_test))
    # print("Accuracy score: {}".format(accuracy_score(y_test, predictions)))
    print("Error rate: {}/{} = {}".format(error, len(Y_test),stats["Error"]))
    print("Accuracy: {}/{} = {}".format(correct, len(Y_test), stats["Accuracy"]))
    
    TP = sum([pp[0] == pp[1] and pp[1] for pp in zip(predictions, Y_test)])
    TN = sum([pp[0] == pp[1] and not pp[1] for pp in zip(predictions, Y_test)])
    FP = sum([pp[0] != pp[1] and pp[0] for pp in zip(predictions, Y_test)])
    FN = sum([pp[0] != pp[1] and not pp[0] for pp in zip(predictions, Y_test)])
    stats["TP"] = TP
    stats["TN"] = TN
    stats["FP"] = FP
    stats["FN"] = FN
    #stats["Precision"] = TP / float(TP + FP)
    #stats["Recall"] = TP / float(TP + FN)
    stats["Precision"] = metrics.precision_score(Y_test, predictions, average='weighted' if weighted else 'binary')
    stats["Recall"] = metrics.recall_score(Y_test, predictions, average='weighted' if weighted else 'binary')
    #stats["F1"] = 2 / (1 / (TP / float(TP + FP)) + 1 / (TP / float(TP + FN)))
    stats["F1"] = metrics.f1_score(Y_test, predictions, average='weighted' if weighted else 'binary')
    try:
        stats["AUC"] = roc_auc_score(Y_test, prediction_probs[:, 1], average="weighted" if weighted else 'macro')
    except:
        stats["AUC"] = 0.0
    stats["F1_micro"]=metrics.f1_score(Y_test, predictions, average='micro')

    print("TP : ", TP)
    print("TN : ", TN)
    print("FP : ", FP)
    print("FN : ", FN)
    print("Diff :", stats["Diff"])
    print("Precision: {}/{} = {}".format(TP, TP + FP, stats["Precision"]))
    print("Recall   : {}/{} = {}".format(TP, TP + FN, stats["Recall"]))
    print("F1 score :       = {}".format(stats["F1"]))
    print("F1_micro score:       = {}".format(stats["F1_micro"]))
    print("AUC: = {}".format(stats["AUC"]))

    preds=dict()
    for ind in range(0, len(X_test.index)):
        preds[X_test.index[ind]]=[predictions[ind], Y_test[ind]]

    with open('results/'+args.saveas.split('/')[-1]+'_preds.txt', 'w') as res:
        s=json.dumps(preds, default=default)
        #s=jsonpickle.encode(d)
        res.write(s)
        
    with open('models/' + args.saveas.split('/')[-1]+'.pickle', 'wb') as m:
        pickle.dump(clf, m)
        print("saved model")
        
    if allpreds:
        print("hp-pred> {0}".format(",".join([str(x) for x in predictions])))
        print("hp-y> {0}".format(",".join([str(x) for x in Y_test])))
    if args.tfpn is not None:
        tfpn = pd.DataFrame(list(zip(predictions, Y_test, X_test.index)), columns=['prediction', 'Y', 'program'])
        tfpn.set_index("program", inplace=True)
        # tfpn["TFPN"] = ""
        tfpn.loc[(tfpn["prediction"] == tfpn["Y"]) & tfpn["prediction"], "TFPN"] = "TP"
        tfpn.loc[(tfpn["prediction"] == tfpn["Y"]) & (tfpn["prediction"] == 0), "TFPN"] = "TN"
        tfpn.loc[(tfpn["prediction"] != tfpn["Y"]) & tfpn["prediction"], "TFPN"] = "FP"
        tfpn.loc[(tfpn["prediction"] != tfpn["Y"]) & (tfpn["prediction"] == 0), "TFPN"] = "FN"
        tfpn.to_csv(args.tfpn, columns=["TFPN"])
        # print(tfpn)

    return stats


def run_majority(X_train, Y_train, X_test, Y_test):
    class pred_majority:
        def __init__(self, X_train, Y_train):
            self.true_per = (sum([x == 1 for x in Y_train]) + 0.0)/ len(Y_train)
            print('true per:' , self.true_per)

        def predict(self, test_set):
            return [np.random.choice([0, 1], p=[1 -self.true_per, self.true_per]) for _ in test_set]

        def predict_proba(self, test_set):
            return np.array([ [1 - self.true_per, self.true_per] for _ in test_set])

    clf=DummyClassifier(strategy='stratified')
    clf.fit(X_train, Y_train)
    return predict(clf, X_test, Y_test)



def run_rf(X_train, Y_train, X_test, Y_test, feature_names, cv=False,  trees=20, validation=False, learning=False,
           gridsearch=False, tree=False, cv_template=False, train_test_indices=None, balance_by_weight=False,
           feature_selection=False, filename=None, test_X=None, test_Y=None):
    # assert not cv or not cv_template
    if feature_selection:
        results = []
        for f in range(5, 100, 100):
            print('Features : ', f)
            clf = RandomForestClassifier(bootstrap=True, n_estimators=50, min_samples_split=6,
                                         criterion='entropy', max_depth=None, n_jobs=-1)
            selector = RFECV(clf, step=1, cv=5, verbose=True, n_jobs=-1, scoring=metrics.make_scorer(metrics.f1_score))
            selector = selector.fit(X_train, Y_train)
            for i in range(len(feature_names)):
                print(feature_names[i],':', selector.support_[i], ', ', selector.ranking_[i])

            X_train_new = selector.transform(X_train)
            X_test_new = selector.transform(X_test)
            print(np.shape(X_train_new))
            print(np.shape(X_test_new))
            #clf.fit(X_train_new, Y_train)
            results.append(predict(selector, X_test, Y_test, allpreds=args.allpreds))
        plot_results(label, results, range(5, 100, 100), xlabels, args.saveas, args)

    elif cv and len(Y_train) > 2:
        best_k, best_score = -1, -1
        clfs = {}
        if cv_template and train_test_indices is not None:
            # do cross validation based on templates
            cv_indices = test_train_validation_splits_by_template(train_test_indices[0], 10, prog_map, args.feature_file)
            print('cv_indices', np.shape(cv_indices))
        else:
            print(sum(Y_train))            
            cv_indices = int(min(5, sum(Y_train), len(Y_train) - sum(Y_train)))
            
        for k in [40, 80, 100, 250, 500]:
            rfconfig = dict()
            rfconfig['n_estimators'] = k
            rfconfig['n_jobs'] = -1
            rfconfig['class_weight'] = 'balanced' if balance_by_weight else None
            #from sklearn import preprocessing
            pipe = Pipeline([['clf', RandomForestClassifier(**rfconfig)]])
            pipe.fit(X_train, Y_train)
            print("X train size::: " + str(len(X_train)))
            
            scores = cross_val_score(pipe, X_train, Y_train, cv=cv_indices, scoring='f1_micro', n_jobs=-1)
            print('rf-n_est={}\nValidation accuracy: {}'.format(k, scores.mean()))
            print(scores)
            if scores.mean() > best_score:
                best_k, best_score = k, scores.mean()
            clfs[k] = pipe
        clf = clfs[best_k]
        print("Best k: ", best_k)
    elif tree and len(Y_train) > 2:
        clfs=dict()
        best_score=0
        best_k=-1
        for k in [10, 40,  80, 100, 250, 500]:
            rfconfig = dict()
            rfconfig['n_estimators'] = k
            rfconfig['n_jobs'] = -1
            rfconfig['class_weight'] = 'balanced' if balance_by_weight else None
            #pipe = Pipeline([['clf', RandomForestClassifier(**rfconfig)]])
            c=RandomForestClassifier(**rfconfig)
            c.fit(X_train, Y_train)

            scores = cross_val_score(c, X_train, Y_train, cv=5, scoring='f1_micro', n_jobs=5)
            print('rf-n_est={}\nValidation accuracy: {}'.format(k, scores.mean()))
            print(scores)
            if scores.mean() > best_score:
                best_k, best_score = k, scores.mean()
            clfs[k] = c
        clf = clfs[best_k]

        arr = X_test.loc[args.special_index]
        prediction, bias, contributions = ti.predict(clf, np.array([arr]))
        for i in range(len(prediction)):
            print("Instance", i)
            print("Bias (trainset mean)", bias[i])
            print("Prediction :", prediction[i])

            print("Top 20 Features:")
            # print("motif:count :: [not converging contribution, converging contribution]")
            for c, feature, count in sorted(zip(contributions[i],
                                         feature_names, arr),
                                     key=lambda x: -abs(x[0][1]))[:20]:

                print("{0}:{2} :: {1}".format(feature_to_ast(feature), c, int(count)))
            print("-" * 20)
        predict(clf, X_test, Y_test)
        exit(0)
    elif learning and len(Y_train) > 2:
        clf = RandomForestClassifier(n_estimators=trees, class_weight='balanced' if balance_by_weight else None, n_jobs=-1)
        plot_learning_curve(clf, 'abc', X_train, Y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=9)
        plt.show()
        exit(1)

        clf = RandomForestClassifier(n_estimators=trees, class_weight='balanced' if balance_by_weight else None, n_jobs=-1)
        print(len(X_train))
        print(len(Y_train))
        train_sizes, train_scores, valid_scores= learning_curve(clf, X_train, Y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
        print(train_sizes)
        print(train_scores)
        print(valid_scores)
        plt.plot(train_sizes, [np.mean(x) for x in train_scores], label='training')
        plt.plot(train_sizes, [np.mean(x) for x in valid_scores], label='validation')
        plt.legend()
        plt.show()
        exit(1)

    elif validation and len(Y_train) > 2:
        train_sizes = np.linspace(80, 250, 10, dtype=np.int)
        train_scores, valid_scores = validation_curve(RandomForestClassifier(), X_train, Y_train, "n_estimators", train_sizes, cv=10)
        plt.plot(train_sizes, [np.mean(x) for x in train_scores], label='training')
        plt.plot(train_sizes, [np.mean(x) for x in valid_scores], label='validation')
        plt.legend()
        plt.show()
        exit(1)
    elif gridsearch and len(Y_train) > 2:
        param_dist = {"max_depth": list(range(10, 100, 10)) + [None],
                      "max_features": ['auto', 10, 20, 30, 50, 80, 100],
                      "min_samples_split": np.linspace(2, 20, 5, dtype=np.int),
                      "bootstrap": [True, False],
                      "min_samples_leaf": [ 1, 2, 5, 10],
                      "n_estimators": [20, 50, 100, 250, 500]

                      }
        if cv_template and train_test_indices is not None:
            # do cross validation based on templates
            cv_indices = test_train_validation_splits_by_template(train_test_indices[0], 5, prog_map)
            print('cv_indices', np.shape(cv_indices))
        else:
            cv_indices = int(min(4, sum(Y_train), len(Y_train) - sum(Y_train)))
            
        grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(class_weight='balanced' if balance_by_weight else None),
                                         param_distributions=param_dist,
                                         n_iter=500,
                                         cv=cv_indices,
                                         verbose=2,
                                         n_jobs=-1,
                                         scoring='f1_micro'
                                         )
        grid_search.fit(X_train, Y_train)
        print(grid_search.best_params_)
        best_grid = grid_search.best_estimator_
        predict(best_grid, X_test, Y_test)
        exit(1)
    else:
        rfconfig=None
        if rfconfig is not None:
            print(rfconfig)
            clf = RandomForestClassifier(**rfconfig)
        else:
            clf = RandomForestClassifier(n_estimators=50, class_weight='balanced' if balance_by_weight else None, n_jobs=-1)
        if args.with_noise:
            ###################################Noisy
            from cleanlab.classification import LearningWithNoisyLabels
            from cleanlab.noise_generation import generate_noisy_labels, generate_noise_matrix_from_trace
            from cleanlab.util import print_noise_matrix

            ##Generate Noisy Label
            for noise_level in [0.01,0.02,0.03,0.04,0.05]:
                print("Noise level: {}".format(noise_level))
                np.random.seed(seed=42)
                py = np.bincount([0 if yy < 0.5 else 1 for yy in Y_train]) / float(len(Y_train))
                noise_matrix = generate_noise_matrix_from_trace( K = 2, trace = 2 * (1-noise_level) , py = py, frac_zero_noise_rates = 0)
                print_noise_matrix(noise_matrix)
                Y_noisy_train = generate_noisy_labels(Y_train, noise_matrix)

                print("Fitting with 6s RF")
                # Fit with normal RF
                best_k=-1
                best_score=0
                clfs=dict()
                for k in [40, 80, 100, 250, 500]:
                    rfconfig = dict()
                    rfconfig['n_estimators'] = k
                    rfconfig['n_jobs'] = -1
                    rfconfig['class_weight'] = 'balanced' if balance_by_weight else None
                    pipe=RandomForestClassifier(**rfconfig)
                    pipe.fit(X_train, Y_noisy_train)
                    scores = cross_val_score(pipe, X_train, Y_noisy_train, cv=5, scoring='f1_micro', n_jobs=-1)
                    if scores.mean() > best_score:
                        best_k, best_score = k, scores.mean()
                        clfs[k] = pipe
                clf = clfs[best_k]                

                predict(clf, X_test, Y_test)
                
                print("Fitting with cleanlab")
                clf = RandomForestClassifier(n_estimators=best_k, class_weight='balanced' if balance_by_weight else None, n_jobs=-1)
                lnl = LearningWithNoisyLabels(clf=clf)
                lnl.fit(np.array(X_train), np.array([0 if yy < 0.5 else 1 for yy in Y_noisy_train]))
                predict(lnl, X_test, Y_test)
                
            return
        else:
            clf.fit(np.array(X_train), np.array(Y_train))
    try:
        show_coefficients("rf", clf, feature_names=feature_names, top_features=20)
    except:
        pass
    if test_X is not None:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return predict(clf, test_X, test_Y, allpreds=args.allpreds)
    if Y_test is not None:
        return predict(clf, X_test, Y_test,allpreds=args.allpreds)
    else:
        return clf


def clean(f):
    f.index = f.index.map(lambda x: x.split('/')[-2] if len(x.split('/')) > 1 else x)
    for pat in ignore_indices:
        f = f.filter(set(f).difference(set(f.filter(regex=(pat)))))
    f = f.replace('inf', np.inf)
    f = f.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)
    return f

# parse arguments
args = read_args()
prog_map = {}
# read features and labels
features=read_all_csvs(args.feature_file, prog_map, 'program')
labels=read_all_csvs(args.labels_file, prog_map, 'program')

#print(features.index.tolist()[:5])
# keep only the selected indices using lsh
###########################################
if args.selected is not None:
    selected_indices=open(args.selected).read().strip().splitlines()
    selected_indices=list(set(list(labels.index)).intersection(selected_indices))
    print('Before:: ', len(labels))
    labels = labels.loc[selected_indices]
    print('Keeping indices:: ', len(labels))
################################

label=args.metric
threshold=args.threshold
test_train_split = args.split_ratio
algorithm = args.algorithm
plot_table=args.plot
split_by_template=args.split_by_template
split_by_class=args.split_class
ignore_indices=[] if args.ignore_vi else []
ignore_indices_other = []

shuffle=args.shuffle
thresholds={ 'rhat_min' : [1.05, 1.1, 1.15, 1.2]
             }
xlabels = {
           'rhat_min': 'Threshold for Gelman-Rubin Diagnostic',
           }

runtimexlabels = {'kl': 'MCMC Iterations', 'rhat_min' : 'MCMC Iterations', 'klfix': 'MCMC Iterations', 'wass': 'MCMC Iterations'}
if args.plot:
    threshold = thresholds[label]
    if args.threshold:
        threshold=[float(args.threshold)]

# cleaning
features.index = features.index.map(lambda x:  x.split('/')[-2] if len(x.split('/')) > 1 else x)

if args.keep is not None:
    if '_ast_' in args.keep:        
        args.keep = [ k for k in args.keep if k != '_ast_' ] + ["[0-9_]+"]
    
    f_keep = []
    import re
    for k in args.keep:
        f_keep += list(filter(lambda x: re.match(k, x) is not None, list(features)))
    # print(f_keep)

    features=features.filter(f_keep)

for pat in ignore_indices:
    features=features.filter(set(features).difference(set(features.filter(regex=(pat)))))

# join additional features
if args.feature_other is not None:
    # update table with additional features
    for fo in args.feature_other:
        feature_other=read_all_csvs(fo, prog_map, index='program')
        feature_other.index = feature_other.index.map(lambda x:  x.split('/')[-2] if len(x.split('/')) > 1 else x)
        for pat in ignore_indices:
            feature_other = feature_other.filter(set(feature_other).difference(set(feature_other.filter(regex=(pat)))))
        # remove some columns
        for ig in ignore_indices_other:
            feature_other=feature_other.filter(set(feature_other).difference(set(feature_other.filter(regex=(ig)))))

        features=pd.merge(left=features, right=feature_other, how='left', left_index=True, right_index=True, sort=False)


features=features.replace('inf', np.inf)
features=features.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)

if args.metrics_suffix is None:
    metric_label = label + '_result'
    metric_value = label + '_value'
else:
    metric_label = label + '_result_' + args.metrics_suffix
    metric_value = label + '_value_' + args.metrics_suffix

print_stats(labels, metric_value)


def transform(row, motifs):
    nonzeroindices=row[row>0]
    k=list(nonzeroindices.keys())
    if len(k) > motifs:
        columns=np.random.choice(k, len(k)-motifs)
        for c in columns:
            row[c]=0.0
    return row


def overlap(c1, c2):
    for k in range(len(c1)-2, 0, -3):
        if c1[k:] == c2[:len(c1)-k]:
            return True
            
    return False


if args.max_motifs is not None and args.max_motifs > 0:
    print("Removing max motifs.. upto: {0}".format(args.max_motifs))
    features=features.swifter.apply(lambda row: transform(row,args.max_motifs), axis=1)
    features=features.fillna(0)
    

if args.nooverlap is not None:
    print("Choosing non overlapping motifs")
    cols=list(features)
    np.random.shuffle(cols)
    newcols=[cols[0]]
    removed=[]
    for c in cols[1:]:
        o=False
        for p in newcols:
            if overlap(p, c) or overlap(c,p):
                o = True
                break
        if not o:
            newcols.append(c)
        else:
            removed.append(c)
    print("Old columns #", len(cols))
    print("New columns #", len(newcols))
    ##
    newcols=np.random.choice(newcols, int(args.nooverlap*len(newcols)))
    print("Sampled columns #", len(newcols))
    ##
    features=features[newcols]
    print("New columns:", newcols)
    
# shuffling features and labels
if args.allpreds is True:
    print("not permuting!!!")
else:
    features = features.reindex(np.random.permutation(features.index))
labels = labels.reindex(np.random.permutation(labels.index))


if args.metric_filter is not None:
    m=args.metric_filter.split(":")[0]
    mval=float(args.metric_filter.split(":")[1])
    print("Filtering based on metric: {0}, value: {1}".format(m, mval))
    print("Original label size: {0}".format(len(labels)))
    labels = labels[labels[m] <= mval]
    print("Filtered label size: {0}".format(len(labels)))

for i in list(features.index):
    if 'progs20190621-213255446133' in i or 'progs20190621-205423791989' in i:
        #print(list(features))
        features=features.drop(i)

cols=list(features)
np.random.shuffle(cols)
features = features[cols]
test_features=args.test_features
# to test on external set of programs
if test_features is not None and args.test_labels is not None:
    test_features=read_all_csvs(args.test_features, prog_map, 'program')
    test_labels=read_all_csvs(args.test_labels, prog_map, 'program')
    test_features.index=test_features.index.map(lambda x:  x.split('/')[-2] if len(x.split('/')) > 1 else x)
    for pat in ignore_indices:
        if test_features is not None:
            test_features = test_features.filter(set(test_features).difference(set(test_features.filter(regex=(pat)))))
    if f_keep is not None:
        import re
        ast_stuff=list(filter(lambda x: len(re.findall('[a-zA-Z]',x)) ==0, list(test_features)))
        test_features = test_features.filter(f_keep+ast_stuff)
    test_features = test_features.replace('inf', np.inf)
    test_features = test_features.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)

# transform features


def get_class_dist(X_train, X_test):
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0
    train_class=dict()
    test_class=dict()
    for i in list(X_train.index):
        p = prog_map[i]
        if p in train_class:
            train_class[p] += 1
        else:
            train_class[p] = 1
    for i in list(X_test.index):
        p = prog_map[i]
        if p in test_class:
            test_class[p] += 1
        else:
            test_class[p] = 1
    print("Train")
    print(train_class)
    print("Test")
    print(test_class)


if type(threshold) == list:
    results = []
    repetitions = 3
    for t in threshold:
        print("Threshold : {0}============================================".format(t))
        full_table = filter_by_metrics(features, metric_label, metric_value, labels, t)
        if test_features is not None:
            test_features=test_features.reindex(columns=list(features))
            test_features = test_features.replace('inf', np.inf)
            test_features=test_features.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)
            test_full_table = filter_by_metrics(test_features, metric_label, metric_value, test_labels, t)
        else:
            test_full_table = None
        pos = full_table[full_table[metric_label] == 1].index
        neg = full_table[full_table[metric_label] == 0].index
        pos_samples = len(list(pos))
        neg_samples = len(list(neg))
        print("Total Positive samples: " + str(pos_samples))
        print("Total Negative samples : " + str(neg_samples))
        if args.balance:
            ind = balance(full_table, metric_label)
        else:
            ind = list(full_table.index.tolist())
        for _ in range(repetitions):
            if split_by_template:
                X_train, X_test, Y_train, Y_test, train_ind, test_ind = test_train_split_template(
                    full_table,
                    ind,
                    features,
                    metric_label,
                    test_train_split if args.split_template_name is None else args.split_template_name,
                    shuffle,
                    prog_map,
                    args.train_size,
                    args.projection_size
                )
                get_class_dist(X_train, X_test)

            else:
                X = [list(features.loc[i]) for i in ind]
                if algorithm == 'rf_reg' or args.stratify_data:
                    Y = [full_table.loc[i][metric_value] for i in ind]
                else:
                    Y = [full_table.loc[i][metric_label] for i in ind]

                X_train, X_test, Y_train, Y_test = split_dataset(X, Y,
                                                                 label,
                                                                 ratio=test_train_split,
                                                                 shuffle=shuffle,
                                                                 stratify_data=args.stratify_data,
                                                                 threshold=t if algorithm != 'rf_reg' else None)
                train_ind = test_ind = None

            if algorithm == 'rf':
                if test_full_table is not None:
                    print(test_full_table.index.tolist())
                    results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv,
                                          cv_template=args.cv_template, train_test_indices=(train_ind, test_ind),
                                          filename=args.feature_file,
                                          test_X=[list(test_features.loc[i]) for i in list(test_full_table.index.tolist())],
                                          test_Y=[test_full_table.loc[i][metric_label] for i in list(test_full_table.index.tolist())]))
                else:
                    results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv,
                                      cv_template=args.cv_template, train_test_indices=(train_ind, test_ind),
                                      filename=args.feature_file))
            elif algorithm == 'maj':
                results.append(run_majority(X_train, Y_train, X_test, Y_test))

    plot_results(label, results, threshold, xlabels, args.saveas, args)
elif args.plt_template:
    # plt performance by templates
    results = []
    for templates in range(1,11):
        full_table = filter_by_metrics(features, metric_label, metric_value, labels, threshold)
        if args.balance:
            ind = balance(full_table, metric_label)
        else:
            ind = list(full_table.index.tolist())
        X_train, X_test, Y_train, Y_test, train_ind, test_ind = test_train_split_template(full_table,
                                                                                     ind,
                                                                                     features,
                                                                                     metric_label,
                                                                                     templates,
                                                                                     shuffle,
                                                                                          prog_map, args.train_size)

        results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv,
                                      cv_template=args.cv_template, train_test_indices=(train_ind, test_ind)))
    plot_results(metric_label, results, range(1,11), xlabels, args.saveas, args)
    #exit(1)

elif args.runtime:
    runtime_results=[]
    if args.runtime_iteration is not None:
        runtimes = [int(args.runtime_iteration)]        
    elif args.warmup:
        runtimes= [10, 20, 40, 60, 80, 100, 200, 400, 600]
    else:
        runtimes = [10, 20, 40, 60]

    for iteration in runtimes:
        print("Runtime iteration {0}=====================".format(iteration))
        repetitions = 1
        for _ in range(repetitions):
            features_new = features
            if iteration > 0:
                runtime_files=[]
                for file in args.feature_file:
                    print(file)
                    if args.warmup:
                        runtime_files += ['{0}_warmup_runtime_{1}.csv'.format(file.split('_features')[0], iteration)]
                    else:
                        runtime_files += ['{0}_runtime_{1}.csv'.format(file.split('_features')[0], iteration)]
                print(runtime_files)
                features_new = update_features(features_new, runtime_files, ignore_indices_other, prog_map)

            full_table = filter_by_metrics(features_new, metric_label, metric_value, labels, threshold)
            # if iteration > 0:
            #     checkna(full_table)
            pos = full_table[full_table[metric_label] == 1].index
            neg = full_table[full_table[metric_label] == 0].index
            pos_samples = len(list(pos))
            neg_samples = len(list(neg))
            print("Positive samples: " + str(pos_samples))
            print("Negative samples : " + str(neg_samples))
            if args.balance:
                ind = balance(full_table, metric_label)
            else:
                ind = list(full_table.index.tolist())

            if split_by_template:
                X_train, X_test, Y_train, Y_test, train_ind, test_ind = test_train_split_template(full_table,
                                                                                                  ind,
                                                                                                  features_new,
                                                                                                  metric_label,
                                                                                                  test_train_split if args.split_template_name is None else args.split_template_name,
                                                                                                  shuffle,
                                                                                                  prog_map,args.train_size, args.projection_size)
            else:
                X = [list(features_new.loc[i]) for i in ind]
                if algorithm == 'rf_reg' or args.stratify_data:
                    Y = [full_table.loc[i][metric_value] for i in ind]
                else:
                    Y = [full_table.loc[i][metric_label] for i in ind]
                X_train, X_test, Y_train, Y_test = split_dataset(X, Y,
                                                                 label,
                                                                 ratio=test_train_split,
                                                                 shuffle=shuffle,
                                                                 stratify_data=args.stratify_data,
                                                                 threshold=threshold if algorithm != 'rf_reg' else None)

                train_ind = test_ind = None
            if algorithm == 'rf':
                runtime_results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features_new), cv=args.cv,
                                              cv_template=args.cv_template, train_test_indices=(train_ind, test_ind)))
            elif algorithm == 'maj':
                runtime_results.append(run_majority(X_train, Y_train, X_test, Y_test))

    plot_results(label, runtime_results, runtimes, runtimexlabels, args.saveas, args)
    exit(1)

elif args.train_by_size:
    results = []
    train_size=[0.5, 0.6, 0.7, 0.8, 0.9]
    full_table = filter_by_metrics(features, metric_label, metric_value, labels, threshold)
    pos = full_table[full_table[metric_label] == 1].index
    neg = full_table[full_table[metric_label] == 0].index
    pos_samples = len(list(pos))
    neg_samples = len(list(neg))
    print("Positive samples: " + str(pos_samples))
    print("Negative samples : " + str(neg_samples))
    if args.balance:
        common_indices = balance(full_table, metric_label)
    else:
        common_indices = list(full_table.index.tolist())
    import random
    if shuffle:
        random.shuffle(common_indices)
    for size in train_size:
        print("Training size : {0}=====================".format(size))
        X = [list(features.loc[i]) for i in common_indices]
        Y = [full_table.loc[i][metric_label] for i in common_indices]
        print(np.shape(X))
        print(np.shape(Y))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=size,
                                                            test_size=1.0 - size, shuffle=shuffle)
        print("Train {0}".format(len(X_train)))
        print("Test {0}".format(len(X_test)))
        results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv, cv_template=args.cv_template))

    plot_results(label, results, train_size, xlabels, args.saveas, args)
elif args.predict is not None:
    topredict = read_all_csvs(args.predict,prog_map, 'program')
    topredict = clean(topredict)
    full_table = filter_by_metrics(features, metric_label, metric_value, labels, threshold)
    pos = full_table[full_table[metric_label] == 1].index
    neg = full_table[full_table[metric_label] == 0].index
    pos_samples = len(list(pos))
    neg_samples = len(list(neg))
    print("Positive samples: " + str(pos_samples))
    print("Negative samples : " + str(neg_samples))
    # balance_by_template(full_table, metric_label)

    # under sampling
    if args.balance:
        common_indices = balance(full_table, metric_label)
    else:
        common_indices = list(full_table.index.tolist())

    X = [features.loc[i] for i in common_indices]
    if algorithm == 'rf_reg':
        Y = [full_table.loc[i][metric_value] for i in common_indices]
    else:
        Y = [full_table.loc[i][metric_label] for i in common_indices]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=test_train_split,
                                                        test_size=1.0 - test_train_split)
    if algorithm == 'rf':
        clf = run_rf(X_train, Y_train, None, None, list(features), cv=args.cv, cv_template=args.cv_template,
               gridsearch=args.grid, learning=args.learning, validation=args.validation, tree=args.tree,
               train_test_indices=([x.name for x in X_train], [x.name for x in X_test]),
               feature_selection=args.feature_select)
        predictions = clf.predict(topredict)

        print(sum(predictions))
        print(len(predictions) - sum(predictions))

else:

    #exit(1)
    full_table = filter_by_metrics(features, metric_label, metric_value, labels, threshold)
    pos = full_table[full_table[metric_label] == 1].index
    neg = full_table[full_table[metric_label] == 0].index
    pos_samples = len(list(pos))
    neg_samples = len(list(neg))
    print("Positive samples: " + str(pos_samples))
    print("Negative samples : " + str(neg_samples))
    #balance_by_template(full_table, metric_label)

    # under sampling
    if args.balance:
        common_indices = balance(full_table, metric_label)
    else:
        common_indices = list(full_table.index.tolist())

    # split
    if split_by_class is not None:
        import json
        classfile=json.load(open(split_by_class))
        results=[]
        xlabels = []
        for c in classfile:
            print(c)
            X_train, X_test, Y_train, Y_test, train_ind, test_ind = test_train_split_class(full_table,
                                                                                           common_indices,
                                                                                           features,
                                                                                           metric_label,
                                                                                           prog_map,
                                                                                           list(c.keys())[0],
                                                                                           c, args.train_size)
            xlabels.append(str(list(c.keys())[0]))
            results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv,
                                  cv_template=args.cv_template, train_test_indices=(train_ind, test_ind)))

        barwdith = 0.25
        rng = range(1, len(xlabels) + 1)
        plt.bar(rng, [x["Recall"] for x in results], label='Recall', width=barwdith)
        plt.bar([x + barwdith for x in rng], [x["Precision"] for x in results], label='Precision', width=barwdith)
        plt.bar([x + 2*barwdith for x in rng], [x["F1"] for x in results], label='F1', width=barwdith)
        plt.xticks([x + barwdith for x in rng], xlabels, rotation=20)
        plt.grid(True)
        plt.ylabel("Scores")
        plt.legend(loc='center',  bbox_to_anchor=(0.5,1.11),  ncol=3, prop={'size': 16})
        plt.show()
        exit(1)

    elif split_by_template:
        X_train, X_test, Y_train, Y_test, train_ind, test_ind = test_train_split_template(full_table,
                                                                                          common_indices,
                                                                                          features,
                                                                                          metric_value
                                                                                          if algorithm == 'rf_reg'
                                                                                          else metric_label,
                                                                                          test_train_split if args.split_template_name is None else args.split_template_name,
                                                                                          shuffle,
                                                                                          prog_map, args.train_size, args.projection_size)
    else:
        if args.special_index is not None:
            common_indices.remove(args.special_index)
        X = features.loc[common_indices]

        if algorithm == 'rf_reg' or args.stratify_data:
            Y = full_table.loc[common_indices][metric_value]
        else:
            Y = full_table.loc[common_indices][metric_label]

        print(np.shape(X))
        print(np.shape(Y))
        X_train, X_test, Y_train, Y_test = split_dataset(X, Y,
                                                         label,
                                                         ratio=test_train_split,
                                                         shuffle=shuffle,
                                                         stratify_data=args.stratify_data,
                                                         threshold=threshold if algorithm != 'rf_reg' else None)
        if args.special_index is not None:
            X_test=X_test.append(features.loc[args.special_index])
            print('data',features.loc[args.special_index])
            Y_test=Y_test.append(pd.Series({args.special_index: full_table.loc[args.special_index][metric_label]}))

    get_class_dist(X_train, X_test)


    if algorithm == 'rf':
        run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv, cv_template=args.cv_template, gridsearch=args.grid, learning=args.learning, validation=args.validation, tree=args.tree, train_test_indices=(list(X_train.index), list(X_test.index)), feature_selection=args.feature_select)
    elif algorithm == 'maj':
        run_majority(X_train, Y_train, X_test, Y_test)


