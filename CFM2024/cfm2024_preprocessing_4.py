import pandas as pd
import numpy as np
import os


def build_venue_columns(input_file):
    """Derive the venue column list ONCE, from train only, and reuse it
    for both train and test. Recomputing it separately per file is what
    caused the earlier column-mismatch risk."""
    venues = pd.read_csv(input_file, usecols=["venue"])["venue"].unique()
    return [f"venue_{int(v)}_ratio" for v in sorted(venues)]


def preprocess(input_file, output_file, venue_columns, chunk_size=100_000):
    eps = 1e-6

    # raw-currency / raw-count price features that get converted to
    # tick-normalized, scale-free versions after aggregation, then dropped
    price_like_cols = [
        'median_price', 'min_price', 'max_price', 'median_mid_price',
        'mean_best_ask', 'median_best_ask', 'min_best_ask', 'max_best_ask',
        'mean_best_bid', 'median_best_bid', 'min_best_bid', 'max_best_bid',
        'min_spread', 'max_spread', 'std_spread', 'mean_queue_depth',
    ]

    if os.path.exists(output_file):
        os.remove(output_file)  # avoid silently appending to a stale file on rerun

    for df in pd.read_csv(input_file, chunksize=chunk_size):
        df['spread'] = df['ask'] - df['bid']
        df['mid_price'] = (df['ask'] + df['bid']) / 2
        df['relative_spread'] = df['spread'] / np.clip(df['mid_price'], eps, None)
        df['obi'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
        df['is_trade'] = df['trade'].astype(int)
        df['is_cancel'] = ((df['action'] == 'D') & (~df['trade'])).astype(int)
        df.loc[df['action'] == 'A', 'queue_depth_when_new'] = np.abs(df['price'] - df['mid_price'])

        # flux relative to the local book depth it's acting on, instead of
        # an absolute magnitude that drifts with market-wide liquidity
        # trends over the two years
        df['flux_rel'] = np.abs(df['flux']) / (df['bid_size'] + df['ask_size'] + 1)

        new = df.groupby('obs_id').agg(
            median_price=('price', 'median'),
            min_price=('price', 'min'),
            max_price=('price', 'max'),
            median_mid_price=('mid_price', 'median'),
            mean_best_ask=('ask', 'mean'),
            median_best_ask=('ask', 'median'),
            min_best_ask=('ask', 'min'),
            max_best_ask=('ask', 'max'),
            mean_best_bid=('bid', 'mean'),
            median_best_bid=('bid', 'median'),
            min_best_bid=('bid', 'min'),
            max_best_bid=('bid', 'max'),
            std_spread=('spread', 'std'),
            min_spread=('spread', 'min'),
            max_spread=('spread', 'max'),
            median_relative_spread=('relative_spread', 'median'),
            mean_bid_size_log=('bid_size', lambda x: np.mean(np.log1p(x))),
            median_bid_size=('bid_size', 'median'),
            mean_ask_size_log=('ask_size', lambda x: np.mean(np.log1p(x))),
            median_ask_size=('ask_size', 'median'),
            mean_obi=('obi', 'mean'),
            total_trades=('is_trade', 'sum'),
            total_cancels=('is_cancel', 'sum'),
            mean_flux_rel=('flux_rel', 'mean'),          # replaces mean_flux_log_abs
            unique_orders=('order_id', 'nunique'),
            mean_queue_depth=('queue_depth_when_new', 'mean'),
            tick_size=('price', lambda x: (
                x.drop_duplicates()
                 .sort_values()   # <-- bug fix: must sort before diff().
                 .diff()          #     drop_duplicates() alone preserves
                 .abs()           #     order of first appearance, not price
                 .loc[lambda diffs: diffs > 0]   # order, so the old version measured
                 .min()                          # gaps between temporally-adjacent
            )),                                  # distinct prices, not the true
                                                  # minimum spacing between price
                                                  # levels -- it could miss the
                                                  # actual tick size entirely.
            realized_variance=('mid_price', lambda x: (x.diff() ** 2).sum()),
        )
        # top_book_vol dropped: redundant with realized_variance, and it
        # was a top shift-driver (std of raw mid_price leaks price level)

        new['cancel_ratio'] = new['total_cancels'] / 100
        new['trade_frequency'] = new['total_trades'] / 100

        # convert remaining raw-currency features to tick-normalized,
        # scale-free versions, then drop the raw ones. NaN where tick_size
        # is undefined (single unique price in the sequence) -- HGB
        # handles NaN natively, so no imputation needed.
        for col in price_like_cols:
            new[f'{col}_ticks'] = new[col] / new['tick_size']
        new = new.drop(columns=price_like_cols)

        # venue ratios: always reindexed against the externally supplied,
        # fixed venue_columns list, so train and test get identical
        # columns in identical order regardless of which venues actually
        # appear in a given file. Not removing any venues -- test that
        # via ablation at modeling time instead.
        venue_ratios = (
            df.groupby('obs_id')['venue']
              .value_counts(normalize=True)
              .unstack(fill_value=0)
        )
        venue_ratios.columns = [f"venue_{int(v)}_ratio" for v in venue_ratios.columns]
        venue_ratios = venue_ratios.reindex(columns=venue_columns, fill_value=0)

        X = pd.concat([new, venue_ratios], axis=1).reset_index()

        file_exists = os.path.exists(output_file)
        X.to_csv(output_file, mode='a', index=False, header=not file_exists)


if __name__ == '__main__':
    venue_columns = build_venue_columns('X_train_N1UvY30.csv')

    # preprocess('X_train_N1UvY30.csv', 'X_train_processed_5.csv', venue_columns)
    preprocess('X_test_m4HAPAP.csv', 'X_test_processed_5.csv', venue_columns)  # adjust filename if different