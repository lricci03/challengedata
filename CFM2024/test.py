# We are able to distinguish the train set from the test set (roc_auc of 0.87, see train_vs_test.py)
# So to get a validation set more similar to the test set, we take the 0.2 instances that are more similar to the test set

import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

X_train = pd.read_csv('X_train_processed_4.csv', index_col='obs_id')

y_train = pd.read_csv('y_train_or6m3Ta.csv', index_col='obs_id')['eqt_code_cat']
y_train = y_train.reindex(X_train.index)
assert y_train.notna().all()
X_test = pd.read_csv('X_test_processed_4.csv', index_col='obs_id')

assert list(X_train.columns) == list(X_test.columns), (
    set(X_train.columns) ^ set(X_test.columns),   # symmetric difference — what's mismatched
)

assert X_train.index.is_unique
assert X_test.index.is_unique

# ============ probabilities of being in train vs test set ============
X_shift = pd.concat([X_train,X_test])
y_shift = np.array([0]*len(X_train) + [1]*len(X_test))

# Need to compute probabilities with out-of-fold (OOF)
# in this way the model never sees the data it is predicting
# bc we divide the train set based on the predicted probabilities

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probas = cross_val_predict(
    HistGradientBoostingClassifier(), X_shift, y_shift,
    cv=skf, method='predict_proba', n_jobs=-1
)[:, 1]


auc = roc_auc_score(y_shift,oof_probas)
print(auc)


"""
''' use the pythonic way below
# ========= Sorting X_train by predicted probabilities of being in the test set and choosing 0.8 for the new train set 

train_probas_enum = [(idx, item) for idx,item in enumerate(y_train_probas)]

train_probas_enum_sorted = sorted(train_probas_enum, key=lambda proba: proba[1])

sorted_idx = [idx for idx, _ in train_probas_enum_sorted]
threshold = math.floor(0.8*len(sorted_idx))
sorted_idx_80 = sorted_idx[:threshold]

new_train = X_train.iloc[sorted_idx_80,:]
'''

# ========= Pythonic way of sorting ==============
'''
threshold_value = np.percentile(y_train_probas, 80)

train_mask = y_train_probas < threshold_value
new_X_train = X_train[train_mask]
new_y_train = y_train[train_mask]

val_mask = ~ train_mask
new_X_val = X_train[val_mask]
new_y_val = y_train[val_mask]
'''

# ========= fit and predict on new train/val sets =============
'''
hgb_clf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1,
    random_state=42
)

hgb_clf.fit(new_X_train,new_y_train)

y_val_pred = hgb_clf.predict(new_X_val)

print(accuracy_score(new_y_val, y_val_pred))
# 0.5345149253731343
'''

# ============== weights for test instances in fit method ==============
# let p be the predict_proba of the instance
# Then the instance should have weight p/(1-p) * (2/3)/(1/3) = p/(1-p) * 2
# since the ratio of train : test is 2:1. 


probas = oof_probas[:len(X_train)]  # concat order preserved, so this slice is safe

# ===== weights are clipped to avoid infinity if probabilities are near 1
eps = 1e-6
weights = 2 * probas / np.clip(1 - probas, eps, None) # protects by 0 division -> it substitues 0 with eps

lower = np.percentile(weights, 1)
upper = np.percentile(weights, 99)
weights = np.clip(weights, lower, upper)
# weights /= weights.mean()

ess = weights.sum()**2 / (weights**2).sum()
print(ess, len(weights))

# ============== fit and predict with weights for submission ==============
hgb_clf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1,
    random_state=42
)

hgb_clf.fit(X_train,y_train, sample_weight= weights)

y_pred = hgb_clf.predict(X_test)

# Create submission
submission = pd.DataFrame({
    "obs_id": X_test.index,
    "eqt_code_cat": y_pred
})

# Save
submission.to_csv("submission_3.csv", index=False)

# the score is 0.284
"""