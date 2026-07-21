# Use new features from 'X_train_processed_4.csv' to see if the auc is less
# check their importance


import pandas as pd
import numpy as np
import math

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.inspection import permutation_importance

X_train = pd.read_csv('X_train_processed_5.csv', index_col='obs_id')

y_train = pd.read_csv('y_train_or6m3Ta.csv', index_col='obs_id')['eqt_code_cat']
y_train = y_train.reindex(X_train.index)
assert y_train.notna().all()
X_test = pd.read_csv('X_test_processed_5.csv', index_col='obs_id')

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
probas = cross_val_predict(
    HistGradientBoostingClassifier(), X_shift, y_shift,
    cv=skf, method='predict_proba', n_jobs=-1
)[:, 1]


auc = roc_auc_score(y_shift,probas)
print(auc)
# 0.8562002313036532

# ======== Check feature importance ============

# Ensure your inputs are clean pandas structures
X_shift = pd.DataFrame(X_shift)
y_shift = np.array(y_shift)

# Initialize variables
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probas = np.zeros(len(X_shift))
importance_list = []

# Loop through the cross-validation folds manually
for fold, (train_idx, val_idx) in enumerate(skf.split(X_shift, y_shift)):
    # 1. Split data
    X_train, y_train = X_shift.iloc[train_idx], y_shift[train_idx]
    X_val, y_val = X_shift.iloc[val_idx], y_shift[val_idx]

    # 2. Train the model
    clf = HistGradientBoostingClassifier()
    clf.fit(X_train, y_train)

    # 3. Save out-of-fold predicted probabilities
    oof_probas[val_idx] = clf.predict_proba(X_val)[:, 1]

    # 4. Calculate Permutation Feature Importance on the validation fold
    # This evaluates how much your metric drops when a feature is shuffled
    result = permutation_importance(
        clf,
        X_val,
        y_val,
        scoring="roc_auc",  # Metric matched to your evaluation
        n_repeats=5,  # Repeats per feature for stability
        random_state=42,
        n_jobs=-1,
    )

    # 5. Store the raw importances
    importance_list.append(result.importances_mean)

# --- Process and View the Global Feature Importances ---

# Average the importances across all 5 cross-validation folds
mean_importances = np.mean(importance_list, axis=0)

# Create a clean summary DataFrame
feature_importance_df = pd.DataFrame(
    {"Feature": X_shift.columns, "Importance_ROC_AUC_Drop": mean_importances}
).sort_values(by="Importance_ROC_AUC_Drop", ascending=False)

print(feature_importance_df)
'''
                   Feature  Importance_ROC_AUC_Drop
10               tick_size                 0.050608
15         min_price_ticks                 0.035693
16         max_price_ticks                 0.032030
35           venue_5_ratio                 0.026256
29  mean_queue_depth_ticks                 0.022930
33           venue_3_ratio                 0.020858
31           venue_1_ratio                 0.018678
27        max_spread_ticks                 0.018311
32           venue_2_ratio                 0.011165
2          median_bid_size                 0.010519
21      max_best_ask_ticks                 0.010298
4          median_ask_size                 0.009712
1        mean_bid_size_log                 0.008025
9            unique_orders                 0.007725
8            mean_flux_rel                 0.007279
11       realized_variance                 0.006519
3        mean_ask_size_log                 0.005833
30           venue_0_ratio                 0.004991
34           venue_4_ratio                 0.003107
24      min_best_bid_ticks                 0.002140
26        min_spread_ticks                 0.002111
20      min_best_ask_ticks                 0.001467
7            total_cancels                 0.000701
19   median_best_ask_ticks                 0.000634
6             total_trades                 0.000499
28        std_spread_ticks                 0.000471
18     mean_best_ask_ticks                 0.000457
23   median_best_bid_ticks                 0.000454
17  median_mid_price_ticks                 0.000255
22     mean_best_bid_ticks                 0.000254
14      median_price_ticks                 0.000250
25      max_best_bid_ticks                 0.000050
0   median_relative_spread                 0.000005
5                 mean_obi                 0.000001
13         trade_frequency                 0.000000
12            cancel_ratio                 0.000000
'''