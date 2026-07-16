import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier


# ============== Train Set Features ============

X_train = pd.read_csv('X_train_processed.csv', index_col='obs_id')

# ============== Train Set Labels ==============

y_train = pd.read_csv('y_train_or6m3Ta.csv', index_col='obs_id')['eqt_code_cat']


# ============== Test Set =================

X_test = pd.read_csv('X_test_processed.csv', index_col='obs_id')

print(X_train.shape)
print(X_test.shape)

print(X_train.columns.equals(X_test.columns))

print(X_train.columns.symmetric_difference(X_test.columns))

'''
# ============== Classification ==============

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

# Create submission
submission = pd.DataFrame({
    "obs_id": X_test.index,
    "eqt_code_cat": y_pred
})

# Save
submission.to_csv("submission.csv", index=False)'''