import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

# Load processed features (no obs_id)
X = pd.read_csv("X_train_processed_3.csv")

# Load labels
y = pd.read_csv(
    "y_train_or6m3Ta.csv",
    index_col="obs_id"
)["eqt_code_cat"]

# Ensure same length
assert len(X) == len(y)

# 66/34 chronological split
split = int(len(X) * 0.663)

X_train = X.iloc[:split]
X_val   = X.iloc[split:]

y_train = y.iloc[:split]
y_val   = y.iloc[split:]

print("Train observations:", len(X_train))
print("Validation observations:", len(X_val))

model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1,
    random_state=42
)

drop_features = [
    "mean_queue_depth",
    "mean_flux_abs"
]

X_train2 = X_train.drop(columns=drop_features)
X_val2 = X_val.drop(columns=drop_features)

model.fit(X_train2, y_train)

pred2 = model.predict(X_val2)

print(
    accuracy_score(y_val, pred2)
)
'''

from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,
    X_val,
    y_val,
    n_repeats=3,
    random_state=42,
    n_jobs=-1
)

importance = pd.Series(
    result.importances_mean,
    index=X_val.columns
).sort_values(ascending=False)

print('feature importance', importance)
'''
'''
mean_queue_depth    0.119143
mean_flux_abs       0.100492
unique_orders       0.048730
mean_ask_size       0.041533
mean_bid_size       0.039743
std_spread          0.034576
top_book_vol        0.025017
median_ask_size     0.020582
median_bid_size     0.019696
total_trades        0.010814
total_cancels       0.005118
mean_obi            0.001728
cancel_ratio        0.000000
trade_frequency     0.000000
'''




