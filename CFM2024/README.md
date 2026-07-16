# Challenge Data - CFM 2024

## Files
* `cfm2024.py`: First feature preprocessing and test of various classifiers, working with only a subset of the train test
* `cfm2024_2.py`: Evaluating the best classifier from `cfm2024.py` on the whole given  test (divided in train and valuation)
* `cfm_preprocessing.py`: preprocess the features of 'X_train_N1UvY30.csv' into the new features to create the new csv file 'X_train_processed.csv'
* `cfm_preprocessing.py`: preprocess the features of 'X_test_m4HAPAP.csv' into the new features to create the new csv file 'X_test_processed.csv'
* `cfm2024_test.py`: First submission: fit `HistBoostingClassifier` to the preprocessed features in the train test and predict the test set
* `cfm2024_3.py`: After poor scoring of the first submission, trying to see what was wrong:
    * check if the distribution of train predictions differs from the one of test predictions
    * check if the distribution of validation predicition differs from the one of test predictions
* `cfm_preprocessing_2.py`: After poor scoring of the first submission, we change some of the features, see [Modifications](#modifications)
* `cfm2024_new_features.py`: Same as `cfm2024_2.py` but with the new features. However, many of the new features have value infinity (we divide by zero), so before continueing we have to re-evaluate the choice of new features.
* `cfm2024_analysis.py`: Use Out-of-Fold (OOF) evaluation to train the model. Compare predicted train labels, true train labels and test labels.
* `test.py`: file to use to test code.


## Challenge goals

The aim of this challenge is to attempt to identify from which stock a piece of tick-by-tick exchange data belongs. The problem is thus a classification task. The exchange data includes each atomic update of the order-book giving information about the best bid and offer prices, any trades that may have occurred, orders that have been placed in the book or cancelled. The order-book is also an aggregated order-book that is built out of multiple exchanges from where we can buy and sell the same shares. Although the data seems very non-descript and anonymous, we expect there to be clues in the data that will give away from which stock a piece of data belongs. This might be through the average spread, the typical quantities of shares at the bid or ask, the frequency with which trades occur, the distribution of how trades are split amongst the venues on which the stock is traded etc. there is a lot of information to aid the participant.

## Data description

**X**: The dataset consists of 100 sequential order-book observations. There are 20 observations randomly taken per stock and per day. There are 504 days in the dataset (approximately 2 years) and 24 stocks. This means there are 100 x 20 x 504 x 24 rows of data = 24'192'000. The columns correspond to the following items:
* obs_id: uniquely identifies a sequence of 100 order book events drawn from a random stock inside a random day; **All rows with same obs_id concern the same stock on the same day**
* venue: The exchange on which the event occurs. It can be NASDAQ, BATY, etc. but they are just encoded in the data as integers;
* action: This is type of order-book event that occurred, it can be ‘A’, ‘D’ or ‘U’. A means volume was added to the book in the form of a new order. ‘D’ means an order was deleted from the book and ‘U’ means an order was updated;
* order_id: The exchange data is ‘Level 3’or Market-by-Order, this means that each update provides a unique identifier for the specific order that was affected. It means that we can track the lifetime of an individual order. If it was placed earlier with a ‘A’, we may see it again deleted in the data by the same market participant if we see the same order id occur again with a ‘D’. Note however that the order-ids have been obfuscated somewhat. The first order referenced in any given sequence of data for a particular observation is given the id=0. If order_id 0 is seen again, you will know that it was the same order again that was affected;
* side: The side of the order-book on which the event took place ‘A’ (Ask, sell) or ‘B’ (Bid, buy);
* price: The price of the order that was affected;
* bid: The price of the best bid;
* ask: The price of the best ask;
* bid_size: The volume of orders at the best bid of the aggregated book;
* ask_size: The volume of orders at the best ask of the aggregated book;
* flux: The change to the order-book affected by the event. i.e. if the volume in a level increased or decreased due to the event;
  * 'flux' quantifies liquidity additions (positive) or removals (negative) at a specific price level
* trade: A boolean true or false to indicate whether a deletion or update event was due to a trade (True) or due to a cancellation (False). 

Because the price itself provides such a large clue, we subtract the best bid price for the first event in the sequence of 100 from the ‘price’, ‘bid’ and ‘ask’ fields.

### NOTES on X
- A (add) `action` means that a trader placed a new limit order in the order book.

    Since it just sits in the order book waiting, it has not traded yet -> `trade` will always be False.

    Since it adds volume, the `flux` will always be positive.
    The `flux` can also be positive when there is an update U in `action`.
- If `trade` is True, the `flux` can only be negative: 'trades consume liquidity'

| **`action`** | **`flux`** | **`trade`** | **Market Meaning** |
|:-:|:-:|:-:|:-:|
| **A (Add)** | Positive (+100) | False | A new limit order was placed in the book. |
| **D (Delete)** | Negative (-100) | False | A trader **cancelled** their order. |
| **D (Delete)** | Negative (-100) | True | A market order hit this limit order and **executed a trade**. |
| **U (Update)** | Positive (+50) | False | A trader increased their existing order size. |
| **U (Update)** | Negative (-50) | True or False | A partial fill occurred (True) or size was manually lowered (False). |

#### EXAMPLE
| action | side | price | bid | ask  | bid_size | ask_size | trade | flux |
|--------|------|-------|-----|------|----------|----------|-------|------|
| A      | A    | 0.3   | 0.0 | 0.01 | 100      | 1        | False | 100  |

Trader Added (A) an ask (side A, selling) at price 0.3 of 100 units
| action | side | price | bid | ask  | bid_size | ask_size | trade | flux |
|--------|------|-------|-----|------|----------|----------|-------|------|
| D      | A    | 0.28  | 0.0 | 0.01 | 100      | 1        | False | -100 |

Trader Deleted (D) and ask (side A, selling) at price 0.28 of 100 units

**Y**: The Y of the dataset is the eqt_code_cat. However, for the training set construction this is an integer between 0 and 23 which identifies the particular stock that was affected.
The training set is drawn from one period of time. The same stocks are used again in the test period, but the observations of the market are drawn from a different future period.

## Preprocessing the data
Each 100 rows with the same obs_id represent the same stock, and we want to use pattern in them as a whole to classify the stock.
Therefore we need to *aggregate* each obs_id into new features, which is called **feature engeneering**.

## Feature engeneering

* median price, lowest price, highest price
* median mid price $\frac{bid + ask}{2}$
* (tick size (*price resolution*): minimum non-zero distance between any two consecutive prices)
* mean, median, max and min of `best_ask`
* mean, median, max and min of `best_bid`
* mea, standard deviation and max spread = `ask - bid`
* mean and median of `bid_size` and `ask_size`
* top-of-book volatility: standard deviation of the `mid_price` over the 100 events
* order book imbalance (OBI): $\frac{size_{bid} - size_{ask}}{size_{bid} + size_{ask}}$

    It measures asymmetry between buying and selling, it is normalized between $0$ and $1$.
    Positive: excess buying pressure.
    Negative: excess selling pressure.
* `venue` distribution (use value_counts())
* order cancellation frequency: ratio of `action == D, Trade == False` over total actions
* average queue depth of new orders: when a new order is added (`action == 'A'`) compute `abs(price - mid_price)`

    check whether market orders are deep into the book or inside the spread
* mean order size: mean `flux` (when `action == 'A'`, did not implement this condition)
* trade frequency: ratio of `trade == 'True'`
* number of unique `order_id` present within the 100 events

Other features we could add 
* number of updates for both ask and bid side
* number of deletions for both ask and bid side
* number of trades
* max flux
* min flux
* average trade size

## Train and Valuation sets

We perform the following with 10'000 `obs_id` (i.e. 100*10'000 observations).

We divide the data in train set (0.8) and valuation set (0.2).


## Classifiers

We test the following classifiers:

* `RandomForestClassifier` with `n_estimators =500`
* `GradientBoostingClassifier`: use early stopping to find the best `n_estimators`
* `HistGradientBoostingClassifier`
* `ExtraTreesClassifier`
* `KNeighborsClassifier`: use grid search to find best `n_neighbors` and `weights`: low accuracy score
* `SVC`: low accuracy score
* `AdaBoostClassifier()`: low accuracy score

We test both hard and soft voting on the best classifiers found above. 

Since RandomForest and GradientBoosting perform bettern than ExtraTrees (accuracy of 0.4 vs. 0.3), we test the following:
* all three
* only RandomForest and GradientBoosting
* all three giving double weight to RandomForest and GradientBoosting

We obtain the best accuracy score with soft voting the last option.

**Best accuracy score**: trained on 8'000 observations, evaluated on 2'000 observations, with soft voting of RandomForest, GradientBoosting and ExtraTrees (with weights 2,2,1): **0.419**.

**Remark**: when training on the full csv, there are some missing values of the *average queue depth*, and Gradient Boosting does not support NaN. So we used HistGradientBoosting instead. It performs better.

Re-running the soft voting with HistGradientBoosting instead of GradientBoosting we get an accuracy of: **0.4645**.

The accuracy of HistGradientBoosting alone is higher than the SoftVotingClassifier: **0.57**.

## First Submission

The accuracy score on the valuation set (0.2 of given train set) is **0.57**, however the submission scores **0.28**.

Looking at the distribution of predictions over the train, valuation and test set (see *cfm2024_3.py*), we notice that on the train and valuation set each class is assigned to ~4% of the instances, which is consistent with the fact that there are 24 stocks (1/24 = 0.0417).

On the test set, the predictions are more skewed, varying between 8% and 1%. Notice that the challenge description doesn't specify how the stocks are distributed in the test set, only that the observations come from a future period.

### Modifications

We can try to remove features that are not stable over time, such as:
* mean, median, max, min price
* mean, std, max spread


** New features**
* Instead of mean, median, max, min price:
price_range_pct = (max_price - min_price) / median_price
* Instead of mean, std, max of spread: mean, std, max of 
relative_spread = spread / mid_price
* Instead of mean of best_bid and of best_ask: take mean of: bid_distance = (mid_price - bid) / mid_price, and 
ask_distance = (ask - mid_price) / mid_price
* Instead of bid_size and ask_side take their ratio: bid_size/ask_size (and take mean and std)
* Add cancel_trade_ratio = total_cancels / (total_trades + 1)


## TO DO
* Tune the hyperparameters of HistBoostingClassifier:
    * max_leaf_nodes
    * l2_regularization
    * learning_rate
    * max_iter

    using OOF log loss (check what it is used on the website!) and not accuracy 
* Check whether train and test set are different: how? Train a classifier to distinguish train from test: 
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        # create a dataset:
        # train rows = 0
        # test rows = 1

        X_shift = pd.concat([X_train, X_test])
        y_shift = np.array(
            [0]*len(X_train) + [1]*len(X_test)
        )
    Then fit a classifier on this (with train and val sets), e.g. HistBoostingClassifier, take the predicted probabilities and use as score the AUC = `roc_auc_score` see p.120-123 of ML-book.

    Then: 
    * if auc ~ 0.5: the model cannot distinguish test from train, so probably unusual class predictions are because the class distribution in the hidden test set is different.
    * if auc ~ 0.8: the test set contains patterns not present in the train. Investigate
            ``drift_model.feature_importances_``
    to find which feature cause drift. 
    