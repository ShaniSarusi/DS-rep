# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 16:41:14 2017

@author: awagner
"""

__author__ = 'gwallach'
# installed required: hyperopt, bson, pymongo
import numpy as np
from hyperopt import fmin, tpe, STATUS_OK, Trials, hp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, r2_score
from sklearn.base import is_classifier
from copy import copy
from functools import partial


class BayesianHyperOpt():
    """ Class for hyper parameter optimization over awkward search spaces,
        which may include real-valued, discrete, and conditional dimensions.
        https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
        Parameters
        ----------
        space : dict of nested function expressions, including stochastic expressions
            Current stochastic expression recognized by hyperopt's optimization algorithm:
                hp.choice(label, options)
                    Returns one of the options, which should be a list or tuple
                hp.randint(label, upper)
                    Returns a random integer in the range [0, upper)
                hp.uniform(label, low, high)
                    Returns a value uniformly between low and high
                hp.quniform(label, low, high, q)
                    Returns a value like round(uniform(low, high) / q) * q (suitable for discrete values)
                hp.loguniform(label, low, high)
                    Returns a value drawn according to exp(uniform(low, high)) so that the logarithm of the
                    return value is uniformly distributed
                hp.qloguniform(label, low, high, q)
                    Returns a value like round(exp(uniform(low, high)) / q) * q
                hp.normal(label, mu, sigma)
                    Returns a real value that's normally-distributed with mean mu and standard deviation sigma.
                    When optimizing, this is an unconstrained variable.
                hp.qnormal(label, mu, sigma, q)
                    Returns a value like round(normal(mu, sigma) / q) * q (suitable for unbounded discrete values)
                hp.lognormal(label, mu, sigma)
                    Returns a value drawn according to exp(normal(mu, sigma)) so that the logarithm of the
                    return value is normally distributed
                hp.qlognormal(label, mu, sigma, q)
                    Returns a value like round(exp(normal(mu, sigma)) / q) * q (suitable for discrete values
                    which are bounded from one side)
        estimator : estimator object
            An object of that type is instantiated for each grid point.
            This is assumed to implement the scikit-learn estimator interface. Either estimator needs
            to provide a score function, or scoring must be passed.
        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or a scorer callable object / function with signature scorer
            (estimator, X, y). If None, the score method of
        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross-validation,
                integer, to specify the number of folds.
                An object to be used as a cross-validation generator.
                An iterable yielding train/test splits.
            For integer/None inputs, if y is binary or multi-class, StratifiedKFold used.
            If the estimator is a classifier or if y is neither binary nor multi-class, KFold is used.
        Attributes
        -----------
            grid_scores_ : list of named tuples
                Contains scores for all parameter combinations in param_grid. Each entry corresponds
                to one parameter setting. Each named tuple has the attributes:
                    parameters - a dict of parameter settings
                    mean_validation_score - the mean score over the cross-validation folds
                    cv_validation_scores - the list of scores for each fold
            best_estimator_ : estimator
                Estimator that was chosen by the search, i.e. estimator which gave highest score
                (or smallest loss if specified) on the left out data. Not available if refit=False.
            best_score_ : float
                Score of best_estimator on the left out data.
            best_params_ : dict
                Parameter setting that gave the best results on the hold out data.
    """

    def __init__(self, space, estimator=None, scoring=None, cv=None, max_evals=100, random_state=123):
        self.space = space
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.max_evals = max_evals
        self.random_state = random_state
        self.grid_scores_ = []
        self.best_estimator_ = None
        self.best_score_ = np.inf
        self.best_params_ = None
        self.trials = None

    def _objective(self, params, data, labels):
        estimator, params = _validate_estimator(self.estimator, params)
        scoring = _validate_scoring(self.scoring, estimator)
        score = -cross_val_score(estimator=estimator, X=data, y=labels, scoring=scoring, cv=self.cv).mean()
        print(len(self.trials))
        print(self.best_score_)
        if score < self.best_score_:
            self.best_score_ = copy(score)
            # enables setting string param arguments such as kernel: "rbf" (trials object convert it to integer)
            self.best_params_ = copy(params)
            # enables using space dictionary of multiple estimators
            self.best_estimator_ = copy(estimator.set_params(**params))
        self.grid_scores_.append((params, -score))
        
        return {'loss': score, 'status': STATUS_OK}

    def fit(self, data, labels):
        objective = partial(self._objective, data=data, labels=labels)
        self.trials = Trials()
        fmin(objective, space=self.space, algo=tpe.suggest, trials=self.trials,
             max_evals=self.max_evals)#, rseed=self.random_state)
        # convert from loss to score
        self.best_score_ = -min(self.trials.losses())


def _validate_estimator(estimator, params):
    if estimator is None:
        estimator = params['estimator']
        del params['estimator']
    estimator.set_params(**params)
    return estimator, params


def _validate_scoring(scoring, estimator):
    if scoring is None:
        # according to sklearn convention - accuracy for classifier and r2 for regressor.
        # hyperopt uses loss instead of accuracy so we negate self.scoring by greater_is_better=False
        if is_classifier(estimator):
            scoring = 'accuracy'
        else:
            scoring = 'r2'
    return scoring