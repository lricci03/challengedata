import pandas as pd
import numpy as np
import os

# =========== Preprocessing ==================

input_file = 'X_test_m4HAPAP.csv'
output_file = 'X_new_test_processed.csv'
chunk_size = 100000


# Get venue categories once
all_venues = pd.read_csv(input_file, usecols=["venue"])["venue"].unique()
venue_columns = [f"venue_{int(v)}_ratio" for v in sorted(all_venues)]


# ---------- helper functions ----------------

def action_transition_features(group):

    actions = group['action'].astype(str).values

    transitions = pd.Series(
        zip(actions[:-1], actions[1:])
    ).value_counts(normalize=True)

    output = {}

    for a1 in ['A','D','U']:
        for a2 in ['A','D','U']:
            key = f"{a1}_to_{a2}"
            output[key] = transitions.get((a1,a2),0)

    return pd.Series(output)



def order_lifetime_features(group):

    # first occurrence of order
    first_seen = group.groupby('order_id').cumcount()

    # number of times each order appears
    counts = group['order_id'].value_counts()

    return pd.Series({

        "unique_orders":
            group['order_id'].nunique(),

        "order_reuse_ratio":
            (counts > 1).mean(),

        "mean_order_events":
            counts.mean(),

        "max_order_events":
            counts.max()
    })


# =========== Processing ======================

for df in pd.read_csv(input_file, chunksize=chunk_size):

    # ---------- basic transformations ----------

    df['spread'] = df['ask'] - df['bid']

    df['mid_price'] = (
        df['bid'] + df['ask']
    ) / 2


    # order book imbalance

    denominator = (
        df['bid_size'] + df['ask_size']
    ).replace(0,np.nan)

    df['obi'] = (
        df['bid_size'] - df['ask_size']
    ) / denominator


    # relative price distance

    df['relative_price'] = (
        df['price'] - df['mid_price']
    )


    # normalized flux

    df['normalized_flux'] = (
        df['flux'] /
        (df['bid_size'] + df['ask_size']).replace(0,np.nan)
    )


    # event indicators

    df['is_trade'] = df['trade'].astype(int)

    df['is_cancel'] = (
        (df['action']=='D') &
        (~df['trade'])
    ).astype(int)


    df['is_add'] = (
        df['action']=='A'
    ).astype(int)

    df['is_delete'] = (
        df['action']=='D'
    ).astype(int)

    df['is_update'] = (
        df['action']=='U'
    ).astype(int)


    # side indicators

    df['is_bid'] = (
        df['side']=='B'
    ).astype(int)

    df['is_ask'] = (
        df['side']=='A'
    ).astype(int)



    # =========== Aggregate features ============

    features = df.groupby('obs_id').agg(

        # ----- spread -----

        mean_spread=('spread','mean'),
        median_spread=('spread','median'),
        std_spread=('spread','std'),
        max_spread=('spread','max'),


        # ----- volatility -----

        mid_price_volatility=(
            'mid_price','std'
        ),

        mid_price_range=(
            'mid_price',
            lambda x: x.max()-x.min()
        ),


        # ----- depth -----

        mean_bid_depth=(
            'bid_size','mean'
        ),

        std_bid_depth=(
            'bid_size','std'
        ),

        mean_ask_depth=(
            'ask_size','mean'
        ),

        std_ask_depth=(
            'ask_size','std'
        ),


        # ----- imbalance -----

        mean_imbalance=(
            'obi','mean'
        ),

        imbalance_volatility=(
            'obi','std'
        ),


        # ----- order flow -----

        add_ratio=(
            'is_add','mean'
        ),

        delete_ratio=(
            'is_delete','mean'
        ),

        update_ratio=(
            'is_update','mean'
        ),


        trade_ratio=(
            'is_trade','mean'
        ),

        cancel_ratio=(
            'is_cancel','mean'
        ),


        # ----- side imbalance -----

        bid_event_ratio=(
            'is_bid','mean'
        ),

        ask_event_ratio=(
            'is_ask','mean'
        ),


        # ----- flux -----

        mean_flux=(
            'flux','mean'
        ),

        flux_std=(
            'flux','std'
        ),

        mean_abs_flux=(
            'flux',
            lambda x: np.mean(np.abs(x))
        ),

        positive_flux_ratio=(
            'flux',
            lambda x: np.mean(x>0)
        ),

        negative_flux_ratio=(
            'flux',
            lambda x: np.mean(x<0)
        ),


        # ----- normalized price position -----

        mean_relative_price=(
            'relative_price','mean'
        ),

        std_relative_price=(
            'relative_price','std'
        ),


        # ----- normalized flux -----

        mean_normalized_flux=(
            'normalized_flux','mean'
        ),

        std_normalized_flux=(
            'normalized_flux','std'
        )


    )


    # ---------- order lifetime statistics ----------

    order_features = (
        df.groupby('obs_id')
        .apply(order_lifetime_features)
    )


    # ---------- action transition matrix ----------

    transition_features = (
        df.groupby('obs_id')
        .apply(action_transition_features)
    )


    # ---------- venue ratios ----------

    venue_ratios = (
        df.groupby('obs_id')['venue']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    venue_ratios.columns = [
        f"venue_{v}_ratio"
        for v in venue_ratios.columns
    ]

    venue_ratios = venue_ratios.reindex(
        columns=venue_columns,
        fill_value=0
    )


    # ---------- combine everything ----------

    X = pd.concat(
        [
            features,
            order_features,
            transition_features,
            venue_ratios
        ],
        axis=1
    ).reset_index()



    # ---------- save ----------

    file_exists = os.path.exists(output_file)

    X.to_csv(
        output_file,
        mode="a",
        index=False,
        header=not file_exists
    )