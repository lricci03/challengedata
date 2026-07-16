import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier


# In cfm2024.py we tested various classifiers on a fraction of the train set
# Here we evaluate the best results from cfm2024.py on the whole train/val set.


# ============== Features ============

X = pd.read_csv("X_train_processed.csv", index_col="obs_id")

# ============== Labels ==============

y = pd.read_csv("y_train_or6m3Ta.csv", index_col='obs_id')['eqt_code_cat']

# check that indices match
# print(X.index.equals(y.index))

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
    weights=[2,2,1],
    voting= 'soft'
)

soft_voting_clf.fit(X_train,y_train)

for name, clf in soft_voting_clf.named_estimators_.items():
    print(name, 'score: ', clf.score(X_val,y_val))

print('soft voting clf score: ', soft_voting_clf.score(X_val,y_val))

# rnd score:  0.48429726368159204
# hgb score:  0.5701181592039801
# extra score:  0.4128109452736318
# soft voting clf score:  0.5644589552238806