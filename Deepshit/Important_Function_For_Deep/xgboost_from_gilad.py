# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:51:14 2016

@author: awagner
"""


from abc import ABCMeta, abstractmethod
from xgboost.sklearn import XGBClassifier, XGBRegressor
from xgboost import plot_importance
import numpy as np
class XGBoost(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, data, labels):
        pass

    @abstractmethod
    def predict(self, data):
        pass

class XGBoost_Classifier2(XGBoost):

    def __init__(self, learning_rate=0.05, n_estimators=100, max_depth=12, min_child_weight=5, gamma=0.6, subsample=0.75,
                        colsample_bytree=1., scale_pos_weight=1,reg_lambda1 = 0, reg_alpha1 = 0):
        self.model = XGBClassifier()
        self.label_name = []
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.reg_alpha1 = reg_alpha1
        self.reg_lambda1 = reg_lambda1
        
    def fit(self, X, y):
        self.model = XGBClassifier(learning_rate=self.learning_rate, n_estimators=self.n_estimators, max_depth=self.max_depth, min_child_weight=self.min_child_weight, gamma=self.gamma, subsample=self.subsample,
                        colsample_bytree=self.colsample_bytree, objective='binary:logistic', nthread=4, scale_pos_weight=self.scale_pos_weight,  reg_alpha = self.reg_alpha1 ,reg_lambda = self.reg_lambda1)
        self.model.fit(X, y)

    def predict(self,data):
        predict_model = self.model.predict(data=data)
        return(predict_model)

    def predict_proba(self, data):
        predict_model_proba = self.model.predict_proba(data=data)
        return(predict_model_proba)

    def set_params(self, **params):
        self.model.set_params(**params)

    def get_params(self):
        self.model.get_params()

class XGBoost_Classifier(XGBoost):

    def __init__(self):
        self.model = None
        self.label_name = []

    def fit(self, X_train, X_test, y_train, y_test):
        self.label_name = y_train.columns
        #self.model = XGBClassifier(learning_rate =0.025, n_estimators=200, max_depth=6, min_child_weight=5, gamma=0.6, subsample=0.55,
        #                colsample_bytree=1., objective= 'binary:logistic', nthread=4, scale_pos_weight=1)
        eval_set = [(X_train, y_train), (X_test, y_test)]
        self.model = XGBClassifier()
        self.model.fit(X=X_train, y=y_train, early_stopping_rounds=20, eval_metric='auc', eval_set=eval_set, verbose=True)
        results = self.model.evals_result()
        plot_importance(self.model)

        # plot classification error
        from matplotlib import pyplot
        epochs = len(results['validation_0']['auc'])
        x_axis = range(0, epochs)
        fig, ax = pyplot.subplots()
        ax.plot(x_axis, results['validation_0']['auc'], label='Train')
        ax.plot(x_axis, results['validation_1']['auc'], label='Test')
        ax.legend()
        pyplot.ylabel('ROC AUC')
        pyplot.title('XGBoost Classification ROC-AUC')
        pyplot.show()

    def predict(self,data):
        predict_model = self.model.predict(data=data)
        return(predict_model)

    def predict_proba(self, data):
        predict_model_proba = self.model.predict_proba(data=data)
        return(predict_model_proba)

