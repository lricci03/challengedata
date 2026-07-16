import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier


# Same as cfm2024_2.py but using the new time-independent features of X_train_processed_2.csv
# Also change weights to the classifiers: 1,2,1 instead of 2,2,1

# ============== Features ============

X = pd.read_csv("X_train_processed_2.csv", index_col="obs_id")

# ============== Labels ==============

y = pd.read_csv("y_train_or6m3Ta.csv", index_col='obs_id')['eqt_code_cat']

# check infinity values in X (we might have divided by 0 in preprocessing)

inf_cols = X.columns[np.isinf(X).any()]

for col in inf_cols:
    print(col, X[col].replace([np.inf, -np.inf], np.nan).isna().sum())

# mean_ask_dist 10022
# mean_bid_dist 10022
# min_rel_spread 46
# max_rel_spread 10020
# mean_bid_ask_ratio 15919
# std_bid_ask_ratio 4866
# price_range_pct 35639


print(X[["median_price", "mid_price", "bid", "ask"]].describe())

'''
# =============== test and validation sets ============

X_train, X_val, y_train, y_val =  train_test_split(X, y, test_size=0.2, random_state=42)

# =============== check for NaN values =============

# print(X.isna().sum().sort_values(ascending=False).head(20))
# mean_queue_depth has 20 NaN


# =============== Classifiers ============

rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
hgb_clf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1,
    random_state=42
)
extra_clf = ExtraTreesClassifier(n_estimators=500, random_state=42)

soft_voting_clf = VotingClassifier(
    estimators=[
        ('rnd', rnd_clf),
        ('hgb', hgb_clf),
        ('extra', extra_clf)  
    ],
    weights=[1,2,1],
    voting= 'soft'
)

soft_voting_clf.fit(X_train,y_train)

for name, clf in soft_voting_clf.named_estimators_.items():
    print(name, 'score: ', clf.score(X_val,y_val))

print('soft voting clf score: ', soft_voting_clf.score(X_val,y_val))

'''