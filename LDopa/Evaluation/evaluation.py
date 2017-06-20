# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:45:20 2017

@author: imazeh
"""

import numpy as np
from sklearn import metrics as met
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt


def per_patient_metrics(final_prediction_df):
    patients_results = {}
    for patient in final_prediction_df.patient.unique():
        if patient not in patients_results:
            patients_results[patient] = {}
        binary_pred = final_prediction_df.binary_prediction[final_prediction_df.patient==patient]
        prob_pred = final_prediction_df.proba_prediction[final_prediction_df.patient==patient]
        label = final_prediction_df.true_label[final_prediction_df.patient==patient]
        patients_results[patient]['conf_matrix'] = met.confusion_matrix(label, binary_pred)
        patients_results[patient]['accuracy'] = met.accuracy_score(label, binary_pred)
        patients_results[patient]['recall'] = met.recall_score(label, binary_pred)
        patients_results[patient]['precision'] = met.precision_score(label, binary_pred)
        try:
            auc = met.roc_auc_score(label, prob_pred)
            patients_results[patient]['auc'] = auc
        except ValueError:
            patients_results[patient]['auc'] = 'NA'
    return patients_results


def global_metrics(final_prediction_df):
    binary_pred = final_prediction_df.binary_prediction
    prob_pred = final_prediction_df.proba_prediction
    label = final_prediction_df.true_label
    print('Confusion matrix:')
    print(met.confusion_matrix(label, binary_pred))
    print('Accuracy:', round(met.accuracy_score(label, binary_pred), 3))
    print('Sensitivity (recall):', round(met.recall_score(label, binary_pred), 3))
    print('Precision:', round(met.precision_score(label, binary_pred), 3))
    try:
        auc = met.roc_auc_score(label, prob_pred)
        print('AUC:', round(auc, 3))
    except ValueError:
        print("Not able to compute AUC since there is only one class in the data")
    return


def per_patient_proportion_correlation(final_prediction_df):
    labels_prop = final_prediction_df[['patient', 'true_label']].groupby('patient').agg('mean')
    labels_prop = labels_prop.true_label
    model_prop = final_prediction_df[['patient', 'binary_prediction']].groupby('patient').agg('mean')
    model_prop = model_prop.binary_prediction
    print ('Per patient proportion correlation:')
    print (pearsonr(labels_prop, model_prop))
    plt.scatter(labels_prop, model_prop)
    plt.plot(np.unique(labels_prop), \
             np.poly1d(np.polyfit(labels_prop, model_prop, 1))(np.unique(labels_prop)),\
             color='r')
    plt.show()
    return

def per_visit_compare_updrs(small_segments_predictions):
    per_visit = small_segments_predictions[['patient', 'prediction_probability', 'visit_date']].groupby(['patient', 'visit_date']).agg(['mean', 'median'])
    per_visit.columns = per_visit.columns.droplevel(0)
    per_visit = per_visit.rename_axis(None, axis=1)
    per_visit.reset_index(inplace=True)
    return per_visit


def create_box_plot(results_plus_updrs, updrs_measure, score_aggregated):
    measure_vals = sorted(results_plus_updrs[updrs_measure].unique().tolist())
    boxes_vals = [np.asarray(results_plus_updrs[score_aggregated][results_plus_updrs[updrs_measure]==x]) for x in measure_vals]
    plt.boxplot(boxes_vals)
    plt.xticks(range(1, len(measure_vals)+1), [str(int(x)) for x in measure_vals])
    return



