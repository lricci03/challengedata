import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone


n_obs = 10000

df = pd.read_csv("X_train_N1UvY30.csv", nrows=100*n_obs)

#print(df.columns) # column labels

# ['obs_id', 'venue', 'order_id', 'action', 'side', 'price', 'bid', 'ask', 'bid_size', 'ask_size', 'trade', 'flux']

#print(df.head())

# Print all orders that appear more than once, and the respective actions
# It turns out (as expected) that if a order appears twice it is first being added (A) and then deleted (D)
'''for i in range(100):
    for j in range(i+1,100):
        order_id_i = df.iloc[i]['order_id']
        order_id_j = df.iloc[j]['order_id']
        if order_id_i == order_id_j:
            print(f'i = {i}, j = {j}')
            print(f'order_id[{i}]= {order_id_i}, order_id[{j}]= {order_id_j}')
            print(f' action {i} = {df.iloc[i]["action"]} , action {j} = {df.iloc[j]["action"]}')
'''


# =========== Preprocessing ==================

# Feature engeneering

# Add columns

df['spread'] = df['ask']- df['bid']
df['mid_price'] = (df['ask'] - df['bid'])/2
df['obi'] = (df['bid_size']-df['ask_size'])/(df['bid_size']+df['ask_size'])
df['is_trade'] = df['trade'].astype(int)
df['is_cancel'] = ((df['action'] == 'D') & (~df['trade'])).astype(int)
# .loc[row_condition, target_column], if condition is not met the value is NaN
df.loc[df['action'] == 'A', 'queue_depth_when_new'] = np.abs(df['price'] - df['mid_price'])



new = df.groupby(['obs_id']).agg(
    median_price=('price','median'),
    min_price=('price','min'),
    max_price=('price','max'),
    median_mid_price=('mid_price','median'),
    mean_best_ask=('ask','mean'),
    median_best_ask=('ask','median'),
    min_best_ask=('ask','min'),
    max_best_ask=('ask','max'),
    mean_best_bid=('bid','mean'),
    median_best_bid=('bid','median'),
    min_best_bid=('bid','min'),
    max_best_bid=('bid','max'),
    std_spread=('spread','std'),
    min_spread=('spread','min'),
    max_spread=('spread','max'),
    mean_bid_size=('bid_size','mean'),
    median_bid_size=('bid_size','median'),
    mean_ask_size=('ask_size','mean'),
    median_ask_size=('ask_size','median'),
    top_book_vol=('mid_price','std'),
    mean_obi=('obi','mean'),
    total_trades=('is_trade','sum'),
    total_cancels=('is_cancel','sum'),
    mean_flux_abs=('flux', lambda x: np.mean(np.abs(x))),
    unique_orders=('order_id','nunique'),
    mean_queue_depth = ('queue_depth_when_new','mean')
    )
new['cancel_ratio'] = new['total_cancels']/100
new['trade_frequency'] = new['total_trades']/100


# VENUE RATIOS

# one row for each obs_id, columns contain the normalized value counts of each value in 'venue'
# fill_value =0: If a venue didn't appear in this sequence, don't leave it blank; fill it with a 0.
venue_ratios = df.groupby(['obs_id'])['venue'].value_counts(normalize=True).unstack(fill_value=0)

# rename the columns as venue_0_ratio, venue_1_ratio, etc
venue_ratios.columns = [f'venue_{col}_ratio' for col in venue_ratios.columns]

# CONCATENATE

X = pd.concat([new, venue_ratios], axis=1)

# ============== Labels ==============

y = pd.read_csv("y_train_or6m3Ta.csv", nrows=n_obs, index_col='obs_id')['eqt_code_cat']

# =============== test and validation sets ============
X_train, X_val, y_train, y_val =  train_test_split(X, y, test_size=0.2, random_state=42)

# ===== DecisionTreeClassifier =========
'''
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train,y_train)
y_val_pred = tree_clf.predict(X_val)
print(f'With {n_obs} observations, the accuracy of tree_clf is: {accuracy_score(y_val, y_val_pred)}')
'''
# With 500 observations, the accuracy of tree_clf is: 0.12
# With 1000 observations, the accuracy of tree_clf is: 0.165
# With 10000 observations, the accuracy of tree_clf is: 0.191

# ====== RandomForestClassifier =========
'''
rnd_clf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators':[100,200,500]}
gridsearch = GridSearchCV(rnd_clf, param_grid=param_grid,cv=3)
gridsearch.fit(X,y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)
'''
# With 10'000 observations the accuracy of rnd_clf with the best_param: n_estimators = 500 is 0.384

# Feature importance with RandomForestClassifier
'''rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(X,y)
for score,name in zip(rnd_clf.feature_importances_,X.columns):
    print(round(score,2),name)'''
# min_price, max_price and mean_flux_abs have 6% importance, all other are below

# ======= AdaBoostClassifier ============
'''
ada_clf = AdaBoostClassifier()
ada_clf.fit(X_train,y_train)
y_val_pred = ada_clf.predict(X_val)
print(f'With {n_obs} observations, the accuracy of ada_clf is: {accuracy_score(y_val, y_val_pred)}')
# With 10000 observations, the accuracy of ada_clf is: 0.11
'''

# ========= GradientBoostingClassifier =======

# Setting a large limit for estimators, but using early stopping to find the sweet spot
'''
gb_clf = GradientBoostingClassifier(
    n_estimators=500, 
    learning_rate=0.1, 
    max_depth=3,
    validation_fraction=0.1, # Uses 10% of training data for validation
    n_iter_no_change=5,     # Stops if validation score doesn't improve for 5 iterations
    tol=0.0001,             # Tolerance for improvement
    random_state=42
)

gb_clf.fit(X_train, y_train)


# Check the exact number of trees the model decided to use
print(f"Optimal number of trees: {gb_clf.n_estimators_}")
# Optimal number of trees: 85
'''
'''
gb_clf = GradientBoostingClassifier(
    n_estimators=85, 
    learning_rate=0.1, 
    max_depth=3,
    random_state=42)
gb_clf.fit(X_train,y_train)
y_val_pred = gb_clf.predict(X_val)
print(f'With {n_obs} observations, the accuracy of gb_clf is: {accuracy_score(y_val, y_val_pred)}')
# With 10000 observations, the accuracy of gb_clf is: 0.4005
'''

# =========== HistGradientBoostingClassifier ============
'''
hgb_clf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1,
    random_state=42
)
hgb_clf.fit(X_train,y_train)
y_val_pred = hgb_clf.predict(X_val)
print(f'With {n_obs} observations, the accuracy of hgb_clf is: {accuracy_score(y_val, y_val_pred)}')
# With 10000 observations, the accuracy of hgb_clf is: 0.4615
'''

# =========== SVC ============
'''
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train,y_train)
y_val_pred = svm_clf.predict(X_val)
print(f'With {n_obs} observations, the accuracy of svm_clf is: {accuracy_score(y_val, y_val_pred)}')
# With 10000 observations, the accuracy of svm_clf is: 0.116
'''

# =========== KNeighborsClassifier =========
'''
knn_clf = KNeighborsClassifier()

param_grid= {'n_neighbors':range(2,20),'weights':['distance','uniform']}
grid_search=GridSearchCV(estimator=knn_clf, param_grid=param_grid,cv=3,scoring="accuracy")
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
# {'n_neighbors': 17, 'weights': 'distance'}
print(grid_search.best_score_)
# 0.11549948792034903
knn_score = grid_search.score(X_val,y_val)
print(f'With {n_obs} observations, the accuracy of knn_clf is: {knn_score}')
# With 10000 observations, the accuracy of knn_clf is: 0.1185

'''

# ============== ExtraTreesClassifier ==========
'''
extra_clf = ExtraTreesClassifier(n_estimators=500, random_state=42)
extra_clf.fit(X_train,y_train)
y_val_pred = extra_clf.predict(X_val)
print(f'With {n_obs} observations, the accuracy of extra_clf is: {accuracy_score(y_val, y_val_pred)}')
# With 10000 observations, the accuracy of extra_clf is: 0.3415
'''

# =========== Hard & Sof Voting ==============

# we don't include svm and knn because their scores are low ~0.11 compared to the rest 0.3-04.
'''
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=85, learning_rate=0.1, max_depth=3, random_state=42)
# svm_clf = SVC(random_state=42)
# knn_clf = KNeighborsClassifier(n_neighbors=17, weights='distance')
extra_clf = ExtraTreesClassifier(n_estimators=500, random_state=42)
'''
'''
hard_voting_clf = VotingClassifier(
    estimators=[
        ('rnd', rnd_clf),
        ('gb', gb_clf),
        ('extra', extra_clf)  
    ]
)

soft_voting_clf = clone(hard_voting_clf)
soft_voting_clf.voting = 'soft'

# soft_voting_clf.set_params(svm__probability=True)

hard_voting_clf.fit(X_train,y_train)
soft_voting_clf.fit(X_train,y_train)

for name, clf in hard_voting_clf.named_estimators_.items():
    print(name, 'score: ', clf.score(X_val,y_val))

print('hard voting clf score: ', hard_voting_clf.score(X_val,y_val))
print('soft voting clf score: ', soft_voting_clf.score(X_val,y_val))

# scores including svm and knn: hard voting clf score:  0.3785, soft voting clf score:  0.4005

# rnd score:  0.405
# gb score:  0.4005
# extra score:  0.3415
# hard voting clf score:  0.398
# soft voting clf score:  0.417
'''

# ===== voting classifiers w/out extra ================
'''
hard_voting_clf = VotingClassifier(
    estimators=[
        ('rnd', rnd_clf),
        ('gb', gb_clf),
    ]
)

soft_voting_clf = clone(hard_voting_clf)
soft_voting_clf.voting = 'soft'

hard_voting_clf.fit(X_train,y_train)
soft_voting_clf.fit(X_train,y_train)

for name, clf in hard_voting_clf.named_estimators_.items():
    print(name, 'score: ', clf.score(X_val,y_val))

print('hard voting clf score: ', hard_voting_clf.score(X_val,y_val))
print('soft voting clf score: ', soft_voting_clf.score(X_val,y_val))

# rnd score:  0.405
# gb score:  0.4005
# hard voting clf score:  0.4075
# soft voting clf score:  0.416
'''

# ===== voting classifiers with weights ================

# rnd and gb perform better (0.4) than extra (0.3) so we weight them more in the VotingClassifier
'''
hard_voting_clf = VotingClassifier(
    estimators=[
        ('rnd', rnd_clf),
        ('gb', gb_clf),
        ('extra', extra_clf)  
    ],
    weights=[2,2,1]
)

soft_voting_clf = clone(hard_voting_clf)
soft_voting_clf.voting = 'soft'

# soft_voting_clf.set_params(svm__probability=True)

hard_voting_clf.fit(X_train,y_train)
soft_voting_clf.fit(X_train,y_train)

for name, clf in hard_voting_clf.named_estimators_.items():
    print(name, 'score: ', clf.score(X_val,y_val))

print('hard voting clf score: ', hard_voting_clf.score(X_val,y_val))
print('soft voting clf score: ', soft_voting_clf.score(X_val,y_val))

# rnd score:  0.405
# gb score:  0.4005
# extra score:  0.3415
# hard voting clf score:  0.4075
# soft voting clf score:  0.419'''

# ========== Soft Voting with weights and HistGB instead of GB ========

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
    voting = 'soft'
)

soft_voting_clf.fit(X_train,y_train)

for name, clf in soft_voting_clf.named_estimators_.items():
    print(name, 'score: ', clf.score(X_val,y_val))

print('soft voting clf score: ', soft_voting_clf.score(X_val,y_val))

# rnd score:  0.405
# hgb score:  0.4615
# extra score:  0.3415
# soft voting clf score:  0.4645