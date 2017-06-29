import numpy as np
from hyperopt import hp

# Parameter search space
space = {}

# One of (1,1), (1,2), or (1,3)
space['vec__ngram_range'] = hp.choice('vec__ngram_range', [(1, 1), (1, 2), (1, 3)])

# Random integer in [1,3]
space['vec__min_df'] = 1 + hp.randint('vec__min_df', 3)

# Uniform between 0.7 and 1
space['vec__max_df'] = hp.uniform('vec__max_df', 0.7, 1.0)

# One of True or False
space['vec__sublinear_tf'] = hp.choice('vec__sublinear_tf', [True, False])

# Random number between 50 and 100
space['kbest__percentile'] = hp.uniform('kbest__percentile', 50, 100)

# Random number between 0 and 1
space['clf__l1_ratio'] = hp.uniform('clf__l1_ratio', 0.0, 1.0)

# Log-uniform between 1e-9 and 1e-4
space['clf__alpha'] = hp.loguniform('clf__alpha', -9 * np.log(10), -4 * np.log(10))

# Random integer in 20:5:80
space['clf__n_iter'] = 20 + 5 * hp.randint('clf__n_iter', 12)


##########################################################################################################
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def objective(params):
    pipe.set_params(**params)
    shuffle = KFold(n_splits=10, shuffle=True)
    score = cross_val_score(pipe, X_train, y_train, cv=shuffle, scoring='roc_auc', n_jobs=1)
    return 1-score.mean()


##########################################################################################################
from hyperopt import fmin, tpe, Trials

# The Trials object will store details of each iteration
trials = Trials()

# Run the hyperparameter search using the tpe algorithm
best = fmin(objective,
            space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)