import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
'''
X_train = pd.read_csv('X_train_processed.csv', index_col='obs_id')

X_test = pd.read_csv('X_test_processed.csv', index_col='obs_id')


# Check whether train and test set are different: how?
# Train a classifier to distinguish train from test
# train rows = 0
# test rows = 1

X_shift = pd.concat([X_train,X_test])
y_shift = np.array([0]*len(X_train) + [1]*len(X_test))

X_shift_train, X_shift_val, y_shift_train, y_shift_val = train_test_split(X_shift,y_shift, test_size=0.2, random_state=42)

hgb_clf = HistGradientBoostingClassifier()
hgb_clf.fit(X_shift_train,y_shift_train)
y_shift_scores_pos = hgb_clf.predict_proba(X_shift_val)[:, 1] # probability of class 1

auc = roc_auc_score(y_shift_val,y_shift_scores_pos)
print(auc)
# 0.8629213443102144

# compute the importance of the features/ which features cause the drift

result = permutation_importance(
    hgb_clf, X_shift_val, y_shift_val, n_repeats=10, random_state=42, scoring="roc_auc"
)

# 3. Format and view the results
importances = pd.DataFrame({
    'feature': X_shift_train.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values(by='importance_mean', ascending=False)

print(importances)
'''

'''
            feature  importance_mean  importance_std
23     mean_flux_abs         0.050881        0.001239
25  mean_queue_depth         0.035211        0.000448
1          min_price         0.033139        0.000609
2          max_price         0.032782        0.000614
19      top_book_vol         0.022757        0.000402
33     venue_5_ratio         0.021236        0.000544
14        max_spread         0.019840        0.000330
31     venue_3_ratio         0.019052        0.000331
29     venue_1_ratio         0.017924        0.000508'''

# In cfm_2024_preprocessing_2.py we changed the features
# mean_flux_abs dividing by top_book_vol
# mean_queue_depth dividing by mid_price (non-zero)
# removed min_price, max_price, median_price

# the new features are stored in X_train_processed_2.csv
# # we try again  


X_train = pd.read_csv('X_train_processed_2.csv', index_col='obs_id')

X_test = pd.read_csv('X_test_processed_2.csv', index_col='obs_id')


# Check whether train and test set are different: how?
# Train a classifier to distinguish train from test
# train rows = 0
# test rows = 1

X_shift = pd.concat([X_train,X_test])
y_shift = np.array([0]*len(X_train) + [1]*len(X_test))

X_shift_train, X_shift_val, y_shift_train, y_shift_val = train_test_split(X_shift,y_shift, test_size=0.2, random_state=42)

hgb_clf = HistGradientBoostingClassifier()
hgb_clf.fit(X_shift_train,y_shift_train)
y_shift_scores_pos = hgb_clf.predict_proba(X_shift_val)[:, 1] # probability of class 1

auc = roc_auc_score(y_shift_val,y_shift_scores_pos)
print(auc)
# 0.8629213443102144

# compute the importance of the features/ which features cause the drift

result = permutation_importance(
    hgb_clf, X_shift_val, y_shift_val, n_repeats=10, random_state=42, scoring="roc_auc"
)

# 3. Format and view the results
importances = pd.DataFrame({
    'feature': X_shift_train.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values(by='importance_mean', ascending=False)

print(importances)