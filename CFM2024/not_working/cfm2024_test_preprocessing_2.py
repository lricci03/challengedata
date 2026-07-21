import pandas as pd
import numpy as np
import os

# =========== Preprocessing ==================

input_file = 'X_test_m4HAPAP.csv'
output_file = 'X_test_processed_2.csv'
chunk_size = 100000

all_venues = pd.read_csv(input_file, usecols=["venue"])["venue"].unique()
venue_columns = [f"venue_{int(v)}_ratio" for v in sorted(all_venues)]

for df in pd.read_csv(input_file, chunksize=chunk_size):
    # Add baseline row metrics
    df['spread'] = df['ask'] - df['bid']
    df['mid_price'] = (df['ask'] + df['bid']) / 2
    df['safe_mid_price'] = np.abs(df['mid_price']) + 1e-6
    df['rel_spread'] = df['spread'] / df['mid_price']
    df['bid_ask_ratio'] = df['bid'] / df['ask']
    df['obi'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
    df['is_trade'] = df['trade'].astype(int)
    df['is_cancel'] = ((df['action'] == 'D') & (~df['trade'])).astype(int)
    
    # Distance normalized cleanly by local safe mid-price
    df.loc[df['action'] == 'A', 'queue_depth_when_new'] = np.abs(df['price'] - df['mid_price']) / df['safe_mid_price']

    new = df.groupby(['obs_id']).agg(
        median_price=('price', 'median'),
        min_price=('price', 'min'),
        max_price=('price', 'max'),
        median_mid_price=('mid_price', 'median'),
        std_rel_spread=('rel_spread', 'std'),
        mid_price_volatility=('mid_price', 'std'),  # Renamed to reflect reality
        mean_obi=('obi', 'mean'),
        total_trades=('is_trade', 'sum'),
        total_cancels=('is_cancel', 'sum'),
        mean_flux_abs=('flux', lambda x: np.mean(np.abs(x))),
        unique_orders=('order_id', 'nunique'),
        mean_queue_depth=('queue_depth_when_new', 'mean')
    )
    
    # 1. Turn absolute prices into relative percentage metrics using the local mid-price
    safe_agg_mid = np.abs(new['median_mid_price']) + 1e-6
    new['min_price_pct'] = (new['min_price'] - new['median_mid_price']) / safe_agg_mid
    new['max_price_pct'] = (new['max_price'] - new['median_mid_price']) / safe_agg_mid
    new['median_price_pct'] = (new['median_price'] - new['median_mid_price']) / safe_agg_mid

    # 2. Add structural behavior ratios
    new['cancel_ratio'] = new['total_cancels'] / 100
    new['trade_frequency'] = new['total_trades'] / 100
    new['cancel_trade_ratio'] = new['total_cancels'] / (new['total_trades'] + 1)
    new['flux_per_order'] = new['mean_flux_abs'] / (new['unique_orders'] + 1)

    # 3. CRITICAL: Reassign the dataframe with the dropped leaking columns!
    new = new.drop(columns=[
        'median_price', 
        'min_price', 
        'max_price', 
        'median_mid_price', 
        'mean_flux_abs'
    ])

    # VENUE RATIOS
    venue_ratios = df.groupby(['obs_id'])['venue'].value_counts(normalize=True).unstack(fill_value=0)
    venue_ratios.columns = [f"venue_{v}_ratio" for v in venue_ratios.columns]
    venue_ratios = venue_ratios.reindex(columns=venue_columns, fill_value=0)

    # CONCATENATE
    X = pd.concat([new, venue_ratios], axis=1).reset_index()

    venue_cols = [f"venue_{i}_ratio" for i in range(6)]

    # Calculate Shannon Entropy (measures routing fragmentation)
    # Adding 1e-8 prevents log(0) errors
    X['venue_entropy'] = -(X[venue_cols] * np.log(X[venue_cols] + 1e-8)).sum(axis=1)

    # Drop the raw drifting venue columns so the model can't use them
    X = X.drop(columns=venue_cols)

    # Save out
    file_exists = os.path.exists(output_file)
    X.to_csv(output_file, mode="a", index=False, header=not file_exists)
