from Gait.Pipeline.StepDetection import StepDetection
import pickle
from os.path import join
import numpy as np
import Gait.config as c
from Utils.DataHandling.data_processing import chunk_it
from hyperopt import fmin, tpe, hp


# split data into k-folds
num_acc=  10
k_folds = 5
all_idx = np.arange(num_acc)
folds_idx = chunk_it(all_idx, k_folds, shuffle=True)
test = []
train = []
test_label = []
train_label = []
for i in k_folds:
    test.append([data[x] for x in folds_idx[i]])
    test.append([label[x] for x in folds_idx[i]])

    tr_idx = np.setdiff1d(all_idx, folds_idx[i])
    train.append([data[x] for x in tr_idx])
    train_label.append([label[x] for x in tr_idx])

# now the data and labels have been split

# define an objective function
def objective(labels, data, win_size, variable_1, variable_2, etc....):
    for i in range(len(data)):
        res = algorithm(data[i], params)
    rmse = xxx
    return rmse

# define a search space
space = {data: data[foldi],
         arg1: np.arange(1,10)
         etc.}
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print best
# -> {'a': 1, 'c2': 0.01420615366247227}
print hyperopt.space_eval(space, best)
# -> ('case 2', 0.01420615366247227}