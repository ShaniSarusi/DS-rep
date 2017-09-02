# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:42:03 2017

@author: imazeh
"""

from sklearn import svm
from sklearn.model_selection import LeaveOneGroupOut
from hyperopt import hp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from Utils.Hyperopt.hyperopt_ATM import BayesianHyperOpt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def optimize_hyper_params(features_df, labels, patients, model_name, hyper_params=None, scoring_measure=None,
                          eval_iterations=20):
    """
    Optimize a model's hyper-parameters, given the features, the labels and the model.
    
    The optimization is done by calling Hyperopt's BayesianHyperOpt.
    Thus, the arguments 'model_name', 'hyper_params', 'scoring_measure' and 'eval_iterations'
    need to fit the specific used function.
    
    Input:
        features_df (Pandas DataFrame): Each column is a feature.
        labels (ndarray): In the size as the number of rows in features_df. Contains the labels.
        patients (pandas Series): In the size as the number of rows in features_df. Contains the patients' IDs.
        model_name (string): The name of the model, e.g. 'svm', 'random_forest'.
        hyper_params (dictionary): Containing the hyper-parameters to be optimized. Each key is the name of the
                               parameter, and the key is the search sapce, defined using hyperopt, for example:
                               {'C': hyperopt.choice('C', [0.1, 1, 10]), 'gamma': hyperopt.uniform('gamma', 0, 5)}.
        scoring_measure (string): The metric used in order to evaluate (and optimize) the hyper-parameters space.
                                  If None, so accuracy is used.
        eval_iterations (int): The number of iterations made, before returning the best scored set of hyper-parameters.

    Output:
        The best estimator model, which can then be used.
    """
    if model_name == 'svm':
        model = svm.SVC(kernel='rbf')
        hyper_params = {'C': hp.choice('C', [0.1, 1, 10]), 'gamma': hp.uniform('gamma', 0, 5)}
    elif model_name == 'random_forest':
        model = RandomForestClassifier(n_jobs=-1)
        hyper_params = {'n_estimators': hp.choice('n_estimators', range(10, 200)),
                        'max_depth': hp.choice('max_depth', [2, 5, 10, 30]),
                        'min_samples_split': hp.choice('min_samples_split', list(range(2, 12, 2))),
                        'min_samples_leaf': hp.choice('min_samples_leaf', list(range(2, 12, 2)))}
    elif model_name == 'random_forest_for_agg':
        model = RandomForestClassifier(n_jobs=-1)
        hyper_params = {'n_estimators': hp.choice('n_estimators', range(10, 500)),
                        'max_depth': hp.choice('max_depth', range(1, 5)),
                        'min_samples_split': hp.choice('min_samples_split', list(range(2, 12, 2))),
                        'min_samples_leaf': hp.choice('min_samples_leaf', list(range(1, 13, 2)))}
    elif model_name == 'xgboost':
        model = XGBClassifier()
        hyper_params = {'learning_rate': hp.uniform('learning_rate', 0, 1),
                        'n_estimators': hp.choice('n_estimators', range(40, 50)),
                        'max_depth': hp.choice('max_depth', range(4, 20)),
                        'min_child_weight': hp.choice('min_child_weight', range(3)),
                        'gamma': hp.uniform('gamma', 0, 1),
                        'subsample': hp.uniform('subsample', 0, 1),
                        'scale_pos_weight': hp.uniform('scale_pos_weight', 0, 5)}
    elif model_name == 'logistic_regression':
        model = LogisticRegression(n_jobs=-1)
        hyper_params = {'penalty': hp.choice('penalty', ['l1', 'l2']),
                        'C': hp.uniform('C', 0, 5),
                        'class_weight': hp.choice('class_weight', ['balanced', None])}
    elif model_name == 'knn':
        model = KNeighborsClassifier()
        hyper_params = {'n_neighbors': hp.choice('n_neighbors', range(1,30)),
                        'p': hp.choice('p', range(1,4))}
    if hyper_params is None:
        print('Cannot train model  - hyper parameters are not defined!')
        return

    # Optimize the hyper-parameters of the model:
    logo = LeaveOneGroupOut()
    cv = logo.split(features_df, labels, patients)
    hyper_opt = BayesianHyperOpt(estimator=model, space=hyper_params, scoring=scoring_measure, cv=list(cv),
                                 max_evals=eval_iterations)
    print("Optimizing the model's hyper parameters...")
    hyper_opt.fit(features_df, labels)
    hyper_opt_model = hyper_opt.best_estimator_
    print('The chosen hyper parameters are:', hyper_opt.best_estimator_)
    return hyper_opt_model


def make_cv_predictions_prob_for_all_segments(features_df, labels, patients, model, task_ids):
    all_preds = []
    patients_id = []
    true_labels = []
    all_tasks = []
    for patient in patients.unique():
        print('making predictions for patient', patient)
        train_features = features_df[np.asarray(patients) != patient]
        train_labels = labels[np.asarray(patients) != patient]
        test_features = features_df[np.asarray(patients) == patient]
        test_labels = labels[np.asarray(patients) == patient]
        tasks = task_ids[np.asarray(patients) == patient]
        # Fit the model and predict:
        model.fit(train_features, train_labels)
        pred = model.predict_proba(test_features)
        all_preds.extend(pred[:, 1])
        true_labels.extend(test_labels)
        patients_id.extend([patient]*len(pred[:, 1]))
        all_tasks.extend(tasks)
    preds_df = pd.DataFrame({'patient': patients_id, 'task': all_tasks, 'prediction_probability': all_preds,
                             'true_label': true_labels})
    return preds_df


def make_cv_predictions_for_agg_segments(aggregated_df, model, binary_class_thresh=0.5):
    patients = []
    binary_preds = []
    prob_preds = []
    labels = []
    features_cols = [x for x in aggregated_df.columns if x not in ['patient', 'true_label', 'task']]
    for patient in aggregated_df.patient.unique():
        print('making predictions for patient', patient)
        train_features = aggregated_df[features_cols][aggregated_df.patient != patient]
        train_labels = aggregated_df['true_label'][aggregated_df.patient != patient]
        test_features = aggregated_df[features_cols][aggregated_df.patient == patient]
        test_labels = aggregated_df['true_label'][aggregated_df.patient == patient]
        # Fit the model and predict:   
        model.fit(train_features, train_labels)
        prob_pred = model.predict_proba(test_features)
        prob_pred = prob_pred[:, 1]
        prob_preds.extend(prob_pred)
        binary_pred = [int(prob >= binary_class_thresh) for prob in prob_pred]
#        binary_pred = model.predict(test_features)
        binary_preds.extend(binary_pred)
        labels.extend(test_labels)
        patients.extend([patient]*len(test_labels))
    preds_df = pd.DataFrame({'patient': patients, 'binary_prediction': binary_preds, 'proba_prediction': prob_preds,
                             'true_label': labels})
    return preds_df
