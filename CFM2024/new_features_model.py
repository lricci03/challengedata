import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Use out-of-fold (OOF) evaluation on new_features
# Divide the train set in e.g. 5 folds, then to predict each fold train on the other ones
# In this way we can use the whole train set to train and get prediction for it that we can score


X_train = pd.read_csv('X_train_processed_2.csv', index_col='obs_id')
y_train = pd.read_csv("y_train_or6m3Ta.csv", index_col='obs_id')['eqt_code_cat']



cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42)

n_classes = len(np.unique(y_train))

oof_pred = np.zeros(
    (len(X_train), n_classes)
)


for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):

    print("Fold:", fold + 1)

    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]

    X_val = X_train.iloc[val_idx]

    hgb_clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1,
        random_state=42
        )
    
    hgb_clf.fit(X_tr, y_tr)

    # predictions for validation fold
    oof_pred[val_idx] = hgb_clf.predict_proba(X_val)


oof_labels = np.argmax(oof_pred, axis=1)
print(accuracy_score(y_train, oof_labels))