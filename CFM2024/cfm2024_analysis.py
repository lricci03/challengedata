import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Instead of dividing the train set into train and valuation
# we use out-of-fold (OOF) evaluation
# Divide the train set in e.g. 5 folds, then to predict each fold train on the other ones
# In this way we can use the whole train set to train and get prediction for it that we can score


X_train = pd.read_csv('X_train_processed.csv', index_col='obs_id')
y_train = pd.read_csv("y_train_or6m3Ta.csv", index_col='obs_id')['eqt_code_cat']
X_test = pd.read_csv('X_test_processed.csv', index_col='obs_id')


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42)

n_classes = len(np.unique(y_train))

oof_pred = np.zeros(
    (len(X_train), n_classes)
)

test_pred = np.zeros(
    (len(X_test), n_classes)
)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):

    print("Fold:", fold + 1)

    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]

    X_val = X_train.iloc[val_idx]

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1,
        random_state=42
        )
    
    model.fit(X_tr, y_tr)

    # predictions for validation fold
    oof_pred[val_idx] = model.predict_proba(X_val)

    # predictions for hidden test set
    test_pred += model.predict_proba(X_test) / cv.n_splits

oof_labels = np.argmax(oof_pred, axis=1)
print(accuracy_score(y_train, oof_labels))
# 0.5684390547263681
print(classification_report(y_train, oof_labels))
'''
           precision    recall  f1-score   support

           0       0.76      0.74      0.75      6700
           1       0.46      0.46      0.46      6700
           2       0.41      0.38      0.39      6700
           3       0.51      0.54      0.52      6700
           4       0.61      0.49      0.54      6700
           5       0.53      0.47      0.50      6700
           6       0.61      0.59      0.60      6700
           7       0.50      0.42      0.46      6700
           8       0.66      0.72      0.69      6700
           9       0.66      0.66      0.66      6700
          10       0.60      0.57      0.59      6700
          11       0.50      0.50      0.50      6700
          12       0.49      0.49      0.49      6700
          13       0.49      0.55      0.52      6700
          14       0.60      0.63      0.62      6700
          15       0.56      0.61      0.58      6700
          16       0.76      0.77      0.77      6700
          17       0.49      0.50      0.50      6700
          18       0.65      0.68      0.66      6700
          19       0.78      0.83      0.81      6700
          20       0.52      0.53      0.52      6700
          21       0.44      0.45      0.44      6700
          22       0.59      0.61      0.60      6700
          23       0.43      0.46      0.45      6700
 accuracy                              0.57    160800
   macro avg       0.57      0.57      0.57    160800
weighted avg       0.57      0.57      0.57    160800
'''

# ========= check the class distribution ===========
# training predictions
print('training predictions', np.bincount(y_train) / len(y_train))
# OOF predictions
print('OOF predictions',np.bincount(oof_labels) / len(oof_labels))
# test predictions
test_labels = np.argmax(test_pred, axis=1)
print('test predictions',np.bincount(test_labels) / len(test_labels))

'''
training predictions [0.04166667 0.04166667 0.04166667 0.04166667 0.04166667 0.04166667
 0.04166667 0.04166667 0.04166667 0.04166667 0.04166667 0.04166667
 0.04166667 0.04166667 0.04166667 0.04166667 0.04166667 0.04166667
 0.04166667 0.04166667 0.04166667 0.04166667 0.04166667 0.04166667]
OOF predictions [0.0408893  0.04095149 0.03837687 0.04438433 0.0332898  0.03674751
 0.04011194 0.035199   0.04546642 0.04179104 0.03972637 0.04210199
 0.04207711 0.0463495  0.04328358 0.04553483 0.04220149 0.04230721
 0.04379975 0.04396766 0.04197139 0.04233831 0.04245647 0.04467662]
test predictions [0.01529412 0.06591912 0.0145098  0.01751225 0.0822549  0.02366422
 0.03682598 0.05132353 0.01904412 0.01877451 0.02966912 0.06560049
 0.04292892 0.05073529 0.05917892 0.06654412 0.0283701  0.05921569
 0.0645098  0.08264706 0.01140931 0.01954657 0.05145833 0.02306373]
 '''

# ========= save predictions in csv ===============

oof_df = pd.DataFrame(
    oof_pred,
    columns=[f"class_{i}" for i in range(n_classes)]
)

oof_df["target"] = y_train.values

oof_df.to_csv("oof_predictions.csv", index=False)


# ========= check OOF confusion matrix ============

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

oof_df = pd.read_csv("oof_predictions.csv")

# recover probabilities
oof_pred = oof_df.drop(columns=["target"]).values

# recover true labels
y_train = oof_df["target"].values

# recover predicted classes
oof_labels = np.argmax(oof_pred, axis=1)

cm = confusion_matrix(y_train, oof_labels)

plt.figure(figsize=(12,10))
plt.imshow(cm)
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()