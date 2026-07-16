import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier


# After the poor score of cfm2024_test.py we try to do some analysis of our method 


# =========== Check how test features differ from train features ========

'''
X_train = pd.read_csv('X_train_processed.csv', index_col='obs_id')

X_test = pd.read_csv('X_test_processed.csv', index_col='obs_id')

features = [
    "median_price",
    "std_spread",
    "mean_obi",
    "total_trades",
    "total_cancels",
    "mean_bid_size",
    "mean_ask_size"
]

comparison = pd.DataFrame({
    "train_mean": X_train[features].mean(),
    "test_mean": X_test[features].mean(),
    "difference_%": (
        (X_test[features].mean() - X_train[features].mean())
        / X_train[features].mean()
    ) * 100
})

print(comparison)
'''
# ====================== Are prediction consistent with train data? ==========
'''
X_train = pd.read_csv("X_train_processed.csv", index_col="obs_id")
X_test = pd.read_csv('X_test_processed.csv', index_col='obs_id')


y_train = pd.read_csv("y_train_or6m3Ta.csv", index_col='obs_id')['eqt_code_cat']

hgb_clf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1,
    random_state=42
)


hgb_clf.fit(X_train,y_train)
y_pred = hgb_clf.predict(X_test)
print(y_train.value_counts(normalize=True))
print(pd.Series(y_pred).value_counts(normalize=True))
'''

# ================ Are predictions on val consistent with predictions on test? =========


X = pd.read_csv("X_train_processed.csv", index_col="obs_id")
y = pd.read_csv("y_train_or6m3Ta.csv", index_col='obs_id')['eqt_code_cat']

X_train, X_val, y_train, y_val =  train_test_split(X, y, test_size=0.2, random_state=42)
X_test = pd.read_csv('X_test_processed.csv', index_col='obs_id')

hgb_clf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1,
    random_state=42
)
hgb_clf.fit(X_train,y_train)

print('valuation test pred', pd.Series(hgb_clf.predict(X_val)).value_counts(normalize=True))
print('test set pred', pd.Series(hgb_clf.predict(X_test)).value_counts(normalize=True))