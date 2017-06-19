import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from os.path import join
import Gait.config as c

with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
with open (join(c.pickle_path, 'features_generic'), 'rb') as fp:
    ft = pickle.load(fp)

# set labels
sample['WalkType'] == 'Regular'
y = (sample['WalkType'] == 'Regular').astype(int)

# set features
x = ft

# split data
person = sample['Person'].unique()

auc = np.zeros(len(person))
for i in range(len(person)):
    train = sample['Person'] != person[i]
    test = train.__invert__()
    skb = SelectKBest(chi2, k=25)
    X_train= skb.fit_transform(abs(x[train].as_matrix()), y[train])  # note the absolute value
    X_test = skb.transform(x[test].as_matrix())

    # train model
    clf = RandomForestClassifier()
    clf.fit(X_train, y[train])

    # eval
    y_scores = clf.predict(X_test)
    auc[i] = roc_auc_score(y[test], y_scores)

# importance = clf.feature_importances_
print('Mean auc is ' + str(auc.mean()))


