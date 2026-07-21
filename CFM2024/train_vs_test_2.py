# Use new features from 'X_train_processed_4.csv' to see if the auc is less
# check their importance


import pandas as pd
import numpy as np
import math

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.inspection import permutation_importance

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
probas = cross_val_predict(
    HistGradientBoostingClassifier(), X_shift, y_shift,
    cv=skf, method='predict_proba', n_jobs=-1
)[:, 1]


auc = roc_auc_score(y_shift,probas)
print(auc)
# 0.8686759345124866

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

'''                   Feature  Importance_ROC_AUC_Drop
24       mean_flux_log_abs                 0.048104
1                min_price                 0.030969
26        mean_queue_depth                 0.030628
2                max_price                 0.027704
36           venue_5_ratio                 0.021338
27               tick_size                 0.019779
32           venue_1_ratio                 0.018781
14              max_spread                 0.018724
20            top_book_vol                 0.017625
34           venue_3_ratio                 0.016455
33           venue_2_ratio                 0.010976
17         median_bid_size                 0.009558
19         median_ask_size                 0.008626
31           venue_0_ratio                 0.007472
25           unique_orders                 0.007423
7             max_best_ask                 0.005088
35           venue_4_ratio                 0.003069
16       mean_bid_size_log                 0.001776
28       realized_variance                 0.001572
10            min_best_bid                 0.001498
18       mean_ask_size_log                 0.001304
6             min_best_ask                 0.001037
13              min_spread                 0.000984
22            total_trades                 0.000918
5          median_best_ask                 0.000671
3         median_mid_price                 0.000566
4            mean_best_ask                 0.000430
23           total_cancels                 0.000412
0             median_price                 0.000290
8            mean_best_bid                 0.000189
12              std_spread                 0.000157
15  median_relative_spread                 0.000045
9          median_best_bid                 0.000044
11            max_best_bid                 0.000036
21                mean_obi                 0.000001
29            cancel_ratio                 0.000000
30         trade_frequency                 0.000000'''