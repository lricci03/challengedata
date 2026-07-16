import pandas as pd
import numpy as np
import os

# =========== Preprocessing ==================

input_file = 'X_test_m4HAPAP.csv'
output_file = 'X_test_processed.csv'
chunk_size = 100000

all_venues = pd.read_csv(input_file, usecols=["venue"])["venue"].unique()
venue_columns = [f"venue_{int(v)}_ratio" for v in sorted(all_venues)]

for df in pd.read_csv(input_file, chunksize=chunk_size):
    # Add columns
    df['spread'] = df['ask']- df['bid']
    df['mid_price'] = (df['ask'] + df['bid'])/2
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

    # Rename numeric columns
    venue_ratios.columns = [f"venue_{v}_ratio" for v in venue_ratios.columns]

    # Add any missing venue columns with 0
    venue_ratios = venue_ratios.reindex(columns=venue_columns,fill_value=0)

    # CONCATENATE

    X = pd.concat([new, venue_ratios], axis=1).reset_index()

    # Check if the file already exists right now
    file_exists = os.path.exists(output_file)

    # Write to CSV w/out re-adding the header after the first chunk
    X.to_csv(output_file, mode="a", index=False, header=not file_exists) # If file exists (True), header=False. If it doesn't (False), header=True.