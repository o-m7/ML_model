"""
build_from_polygon_s3.py

Download raw Polygon S3 quote/tick files for a symbol/date range, aggregate to minute
bars, compute requested quote- and OHLCV-based features, resample to multiple
intraday timeframes (1T/5T/15T/30T), and save per-timeframe parquet files.

Behavior:
- Reads AWS / polygon S3 configuration from .env (or CLI overrides)
- Downloads S3 objects matching prefix and date range into a temp dir
- Reads each file (parquet/csv/json), concatenates and filters by date range
- Aggregates quote-level updates into 1-minute quote bars (OHLCV + quote aggregates)
- Resamples to requested timeframes and computes features (using TA-Lib when available)
- Saves per-timeframe parquet to `out/<symbol>/<symbol>_<TIMEFRAME>.parquet`
- Optionally deletes local raw downloads (default True). Does NOT delete S3 objects

NOTE: This script makes reasonable format assumptions about Polygon export files.
Adjust parsing logic where your raw files have different column names.

Usage example:
python build_from_polygon_s3.py --symbol "C:XAU-USD" --start 2019-01-01 --end 2025-11-25 \
    --timeframes 1T 5T 15T 30T --out feature_store --delete-local

"""

import argparse
import os
from pathlib import Path
import tempfile
import shutil
import boto3
import botocore
import pandas as pd
import numpy as np
import json
import logging
import sys
from datetime import datetime, timezone

# allow using project's feature engineering
sys.path.insert(0, '.')
from citadel_features import StrategyFeatures

try:
    import talib
    HAS_TALIB = True
except Exception:
    HAS_TALIB = False

# dotenv for .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('build_from_polygon_s3')

# default columns mapping expected in quote files
QUOTE_COLUMNS = [
    'timestamp', 'best_bid_price', 'best_ask_price', 'best_bid_size', 'best_ask_size'
]

CORE_COLS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']


def s3_client_from_env(aws_profile=None):
    # Create boto3 client using explicit env vars for Polygon S3.
    # Preferred env vars (as provided):
    # - POLYGON_S3_ACCESS_KEY -> aws_access_key_id
    # - Secret_Access_Key -> aws_secret_access_key
    # - POLYGON_S3_SECRET_KEY -> endpoint URL (Polygon-compatible S3 endpoint)
    # If these are not present, fall back to boto3 session/profile or default credentials.
    access_key = os.getenv('POLYGON_S3_ACCESS_KEY')
    secret_key = os.getenv('Secret_Access_Key')
    endpoint = os.getenv('POLYGON_S3_SECRET_KEY')

    if access_key and secret_key:
        client_kwargs = {
            'aws_access_key_id': access_key,
            'aws_secret_access_key': secret_key,
        }
        if endpoint:
            client_kwargs['endpoint_url'] = endpoint
        try:
            s3 = boto3.client('s3', **client_kwargs)
            return s3
        except Exception as e:
            logger.error('Failed to create boto3 client with provided env vars: %s', e)
            raise

    # Fallback: try creating a session (profile or default credentials)
    session_args = {}
    if aws_profile:
        session_args['profile_name'] = aws_profile
    try:
        session = boto3.session.Session(**session_args)
        s3 = session.client('s3')
        return s3
    except Exception as e:
        logger.error('Failed to create boto3 client via session/profile: %s', e)
        raise


def list_s3_keys(s3, bucket: str, prefix: str, start_date: datetime, end_date: datetime, symbol: str = None):
    # List objects under prefix and filter by last modified date range
    kwargs = {'Bucket': bucket, 'Prefix': prefix}
    keys = []
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get('Contents', []) :
            # If a symbol filter is provided, only include keys that reference that symbol
            if symbol and symbol not in obj.get('Key', ''):
                continue
            lm = obj['LastModified']
            if lm.tzinfo is None:
                lm = lm.replace(tzinfo=timezone.utc)
            if start_date <= lm <= end_date:
                keys.append(obj['Key'])
        if resp.get('IsTruncated'):
            kwargs['ContinuationToken'] = resp.get('NextContinuationToken')
        else:
            break
    logger.info('Found %d candidate S3 keys in %s/%s', len(keys), bucket, prefix)
    return keys


def download_s3_objects(s3, bucket: str, keys, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []
    for k in keys:
        fn = Path(k).name
        dest = out_dir / fn
        if dest.exists():
            logger.debug('Skipping download, file exists: %s', dest)
            local_paths.append(dest)
            continue
        try:
            logger.info('Downloading s3://%s/%s -> %s', bucket, k, dest)
            s3.download_file(bucket, k, str(dest))
            local_paths.append(dest)
        except botocore.exceptions.ClientError as e:
            logger.warning('Failed to download %s: %s', k, e)
    return local_paths


def read_raw_file(path: Path) -> pd.DataFrame:
    # Try reading parquet/csv/json; flexible column name handling
    path = Path(path)
    logger.info('Reading raw file %s', path)
    if path.suffix.lower() == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffix.lower() in ['.csv', '.gz']:
        df = pd.read_csv(path)
    elif path.suffix.lower() in ['.json']:
        df = pd.read_json(path, lines=True)
    else:
        # try parquet as fallback
        df = pd.read_parquet(path)

    # Normalize timestamp
    if 'timestamp' in df.columns:
        # timestamp may be numeric (ns) or ISO strings
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            # fallback: try as epoch ns
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ns')
    elif 'window_start' in df.columns:
        # Polygon minute files often use 'window_start' in ns
        try:
            df['timestamp'] = pd.to_datetime(df['window_start'].astype('int64'), unit='ns')
        except Exception:
            df['timestamp'] = pd.to_datetime(df['window_start'])
    else:
        # try common alternatives
        found = False
        for c in ['t', 'ts', 'time']:
            if c in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df[c])
                except Exception:
                    df['timestamp'] = pd.to_datetime(df[c].astype('int64'), unit='ns')
                found = True
                break
        if not found:
            raise ValueError(f'No timestamp/window_start column found in {path}')

    # Normalize quote columns if present
    # polygon often exports 'bidprice','askprice','bidsize','asksize' etc
    colmap = {}
    mapping_candidates = {
        'best_bid_price': ['bidprice','best_bid','best_bid_price','bid_price','bid_price1'],
        'best_ask_price': ['askprice','best_ask','best_ask_price','ask_price','ask_price1'],
        'best_bid_size': ['bidsize','best_bid_size','bid_size','bid_size1'],
        'best_ask_size': ['asksize','best_ask_size','ask_size','ask_size1'],
    }
    for target, candidates in mapping_candidates.items():
        for c in candidates:
            if c in df.columns:
                colmap[c] = target
                break
    if colmap:
        df = df.rename(columns=colmap)

    # If prices are present but no sizes, fill sizes with 1
    for col in ['best_bid_size','best_ask_size']:
        if col not in df.columns:
            df[col] = 1

    return df


def classify_dataframe(df: pd.DataFrame) -> str:
    """Classify a raw dataframe as 'ohlcv' or 'quotes' or 'unknown'."""
    cols = set(df.columns.str.lower())
    if {'open','high','low','close','volume'}.issubset(cols) and ('ticker' in cols or 'symbol' in cols):
        return 'ohlcv'
    # presence of bid/ask suggests quotes
    if any(c in cols for c in ['best_bid_price','best_ask_price','best_bid','best_ask','bidprice','askprice']):
        return 'quotes'
    return 'unknown'

    return df


def aggregate_quotes_to_minute(df_quotes: pd.DataFrame) -> pd.DataFrame:
    # df_quotes expected to have timestamp, best_bid_price, best_ask_price, best_bid_size, best_ask_size
    df = df_quotes.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df['mid_price'] = (df['best_bid_price'] + df['best_ask_price']) / 2.0
    df['spread'] = df['best_ask_price'] - df['best_bid_price']
    df['rel_spread'] = df['spread'] / (df['mid_price'] + 1e-12)
    df['log_mid'] = np.log(df['mid_price'].replace(0, np.nan)).fillna(method='ffill')

    # Floor timestamps to minute
    df['minute'] = df['timestamp'].dt.floor('T')

    agg_list = []
    groups = df.groupby('minute')
    for minute, g in groups:
        rec = {}
        rec['timestamp'] = minute
        # last snapshot
        last = g.iloc[-1]
        rec['best_bid_price'] = last.get('best_bid_price')
        rec['best_ask_price'] = last.get('best_ask_price')
        rec['best_bid_size'] = last.get('best_bid_size')
        rec['best_ask_size'] = last.get('best_ask_size')
        rec['mid_price'] = last['mid_price']
        rec['spread'] = last['spread']
        rec['rel_spread'] = last['rel_spread']
        rec['log_mid'] = last['log_mid']

        # within-bar aggregates
        rec['mid_high'] = g['mid_price'].max()
        rec['mid_low'] = g['mid_price'].min()
        rec['mid_range'] = rec['mid_high'] - rec['mid_low']
        rec['mid_range_pct'] = rec['mid_range'] / (rec['mid_price'] + 1e-12)
        rec['mid_std'] = g['mid_price'].std()

        rec['spread_max'] = g['spread'].max()
        rec['spread_min'] = g['spread'].min()
        rec['spread_mean'] = g['spread'].mean()
        rec['spread_std'] = g['spread'].std()
        rec['rel_spread_max'] = g['rel_spread'].max()
        rec['rel_spread_mean'] = g['rel_spread'].mean()

        rec['n_quote_updates'] = len(g)
        rec['n_bid_price_changes'] = (g['best_bid_price'].diff() != 0).sum()
        rec['n_ask_price_changes'] = (g['best_ask_price'].diff() != 0).sum()
        rec['n_bid_size_changes'] = (g['best_bid_size'].diff() != 0).sum()
        rec['n_ask_size_changes'] = (g['best_ask_size'].diff() != 0).sum()

        # imbalance metrics per snapshot
        bid = g['best_bid_size']
        ask = g['best_ask_size']
        imbalance = (bid - ask) / (bid + ask + 1e-12)
        rec['imbalance_last'] = imbalance.iloc[-1]
        rec['imbalance_mean'] = imbalance.mean()
        rec['imbalance_std'] = imbalance.std()
        rec['imbalance_max'] = imbalance.max()
        rec['imbalance_min'] = imbalance.min()
        rec['imbalance_change'] = rec['imbalance_last'] - (imbalance.shift(1).iloc[-1] if len(imbalance) > 1 else 0)
        rec['imbalance_sign'] = np.sign(rec['imbalance_last'])

        # notional imbalance
        buy_notional = last['best_bid_price'] * last['best_bid_size']
        sell_notional = last['best_ask_price'] * last['best_ask_size']
        rec['buy_side_notional'] = buy_notional
        rec['sell_side_notional'] = sell_notional
        rec['notional_imbalance'] = (buy_notional - sell_notional) / (buy_notional + sell_notional + 1e-12)

        # microprice
        rec['microprice_last'] = (last['best_bid_price'] * last['best_ask_size'] + last['best_ask_price'] * last['best_bid_size']) / (last['best_bid_size'] + last['best_ask_size'] + 1e-12)
        # within-bar microprice stats
        micro_arr = (g['best_bid_price'] * g['best_ask_size'] + g['best_ask_price'] * g['best_bid_size']) / (g['best_bid_size'] + g['best_ask_size'] + 1e-12)
        rec['microprice_std'] = micro_arr.std()
        rec['microprice_range'] = micro_arr.max() - micro_arr.min()
        rec['microprice_range_pct'] = rec['microprice_range'] / (rec['mid_price'] + 1e-12)

        # direction/updown counts
        rec['n_mid_up'] = (g['mid_price'].diff() > 0).sum()
        rec['n_mid_down'] = (g['mid_price'].diff() < 0).sum()
        rec['n_mid_flat'] = (g['mid_price'].diff() == 0).sum()
        net = rec['n_mid_up'] - rec['n_mid_down']
        rec['net_mid_direction'] = net
        rec['mid_direction_ratio'] = net / (rec['n_mid_up'] + rec['n_mid_down'] + 1e-12)

        # spread reaction counts
        rec['n_spread_widen'] = (g['spread'].diff() > 0).sum()
        rec['n_spread_narrow'] = (g['spread'].diff() < 0).sum()
        rec['spread_widen_ratio'] = rec['n_spread_widen'] / (rec['n_spread_widen'] + rec['n_spread_narrow'] + 1e-12)

        agg_list.append(rec)

    df_min = pd.DataFrame(agg_list)
    # ensure sorted
    df_min = df_min.sort_values('timestamp').reset_index(drop=True)
    return df_min


def resample_ohlcv_from_quotes(df_min: pd.DataFrame) -> pd.DataFrame:
    # Build OHLCV open/high/low/close/volume for minute bars from last snapshot and quote updates
    df = df_min.copy()
    # Use microprice or mid as price for OHLCV close/open? Prefer last mid as close
    df_ohlcv = pd.DataFrame()
    df_ohlcv['timestamp'] = df['timestamp']
    df_ohlcv['open'] = df['mid_price'].shift(1).fillna(method='ffill')
    df_ohlcv['high'] = df[['mid_price','mid_high']].max(axis=1)
    df_ohlcv['low'] = df[['mid_price','mid_low']].min(axis=1)
    df_ohlcv['close'] = df['mid_price']
    # approximate volume by n_quote_updates if real trade volume not available
    df_ohlcv['volume'] = df.get('n_quote_updates', 1)
    return df_ohlcv


def compute_timeframe_features(df_ohlcv: pd.DataFrame, df_quote_min: pd.DataFrame, timeframe: str):
    # Merge OHLCV and quote aggregates, then use StrategyFeatures.add_all_features
    df = df_ohlcv.copy()
    # ensure timestamp index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # resample OHLCV to timeframe
    df_tf = df.resample(timeframe, label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df_tf = df_tf.reset_index()

    # attach quote-derived aggregates aligned to the timeframe by joining on nearest previous minute
    # First ensure minute-indexed quote df
    q = df_quote_min.copy()
    # Ensure quote-minute DF has a 'timestamp' column; if not, make an empty DF so downstream code produces NaNs
    if q.empty or 'timestamp' not in q.columns:
        q = pd.DataFrame(columns=['timestamp'])
        q['timestamp'] = pd.to_datetime(q['timestamp'])
        q = q.set_index('timestamp')
    else:
        q['timestamp'] = pd.to_datetime(q['timestamp'])
        q = q.set_index('timestamp')

    # For each timeframe bar, aggregate the quote-minute rows that fall into that timeframe
    # We will compute aggregated quote stats across the minutes in the TF bar
    tf_agg_list = []
    for _, row in df_tf.iterrows():
        start = row['timestamp']
        end = start + pd.Timedelta(timeframe)
        q_slice = q[start:end - pd.Timedelta('1ms')]
        if q_slice.empty:
            # fill with NaNs or with last known snapshot
            rec = {c: np.nan for c in q.columns}
        else:
            rec = {}
            # take last snapshot within TF
            last = q_slice.iloc[-1]
            for c in ['best_bid_price','best_ask_price','best_bid_size','best_ask_size','mid_price','spread','rel_spread','log_mid','imbalance_last','microprice_last']:
                if c in q_slice.columns:
                    rec[c] = last.get(c)
                else:
                    rec[c] = np.nan
            # aggregates across minutes within TF
            # For simplicity reuse some precomputed minute-level aggregates averaged
            rec['mid_high_tf'] = q_slice['mid_high'].max() if 'mid_high' in q_slice.columns else np.nan
            rec['mid_low_tf'] = q_slice['mid_low'].min() if 'mid_low' in q_slice.columns else np.nan
            rec['mid_range_tf'] = rec['mid_high_tf'] - rec['mid_low_tf']
            rec['mid_range_pct_tf'] = rec['mid_range_tf'] / (rec.get('mid_price', 1) + 1e-12)
            rec['n_quote_updates_tf'] = q_slice['n_quote_updates'].sum() if 'n_quote_updates' in q_slice.columns else 0
            rec['n_mid_up_tf'] = q_slice['n_mid_up'].sum() if 'n_mid_up' in q_slice.columns else 0
            rec['n_mid_down_tf'] = q_slice['n_mid_down'].sum() if 'n_mid_down' in q_slice.columns else 0
            rec['imbalance_mean_tf'] = q_slice['imbalance_mean'].mean() if 'imbalance_mean' in q_slice.columns else np.nan
            rec['imbalance_std_tf'] = q_slice['imbalance_std'].mean() if 'imbalance_std' in q_slice.columns else np.nan
        tf_agg_list.append(rec)

    df_qagg = pd.DataFrame(tf_agg_list)
    df_qagg['timestamp'] = df_tf['timestamp']

    # Avoid duplicate 'timestamp' column when concatenating frame-wise results
    if 'timestamp' in df_qagg.columns:
        df_qagg = df_qagg.drop(columns=['timestamp'])

    df_merged = pd.concat([df_tf.reset_index(drop=True), df_qagg.reset_index(drop=True)], axis=1)

    # Ensure no duplicate column labels remain after concat
    if df_merged.columns.duplicated().any():
        df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

    # Now run StrategyFeatures.add_all_features to produce many OHLCV-based features
    df_features = StrategyFeatures.add_all_features(df_merged, timeframe, required_features_union=None)

    # Use talib for additional indicators if available
    if HAS_TALIB:
        close = df_features['close'].values
        try:
            # example RSI
            df_features['rsi_ta'] = talib.RSI(close, timeperiod=14)
            df_features['ema_50_ta'] = talib.EMA(close, timeperiod=50)
        except Exception:
            pass

    # Fill reasonable NaNs and return
    df_features = df_features.reset_index(drop=True)
    return df_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--end', required=True)
    p.add_argument('--s3-bucket', default=None)
    p.add_argument('--s3-prefix', default=None)
    p.add_argument('--aws-profile', default=None)
    p.add_argument('--timeframes', nargs='+', default=['1T','5T','15T','30T'])
    p.add_argument('--out', default='feature_store')
    p.add_argument('--delete-local', action='store_true', help='Delete local downloaded raw files after processing')
    p.add_argument('--temp-dir', default=None)
    args = p.parse_args()

    # parse dates
    start = pd.to_datetime(args.start).to_pydatetime().replace(tzinfo=timezone.utc)
    end = pd.to_datetime(args.end).to_pydatetime().replace(tzinfo=timezone.utc)

    # Resolve bucket and prefix from environment if not provided on CLI.
    # Accept multiple commonly-used env names; primary bucket env var expected: 'Bucket'
    env_bucket = os.getenv('Bucket') or os.getenv('POLYGON_S3_BUCKET') or os.getenv('BUCKET')
    env_prefix = os.getenv('POLYGON_S3_PREFIX') or os.getenv('Prefix') or os.getenv('POLYGON_S3_PATH') or os.getenv('S3_PREFIX')

    if not args.s3_bucket:
        args.s3_bucket = env_bucket
    if not args.s3_prefix:
        args.s3_prefix = env_prefix

    # If prefix is missing, allow listing the whole bucket by using empty string
    if not args.s3_bucket:
        logger.error(
            'S3 bucket not provided. Set CLI arg --s3-bucket or environment variable "Bucket"'
        )
        return

    if not args.s3_prefix:
        logger.warning('S3 prefix not provided; will list entire bucket (this may be large).')
        args.s3_prefix = ''

    s3 = s3_client_from_env(args.aws_profile)

    # Polygon provides daily aggregate files under known paths. Instead of
    # listing the entire bucket, build the expected daily keys for the
    # minute_agg and quotes paths and attempt to download those files.
    # This avoids downloading unrelated symbols and is much faster.
    def build_daily_keys(start_dt, end_dt, base_prefix):
        keys = []
        for d in pd.date_range(start_dt, end_dt, freq='D'):
            y = d.year
            m = f"{d.month:02d}"
            ds = d.strftime('%Y-%m-%d')
            keys.append(f"{base_prefix}/{y}/{m}/{ds}.csv.gz")
        return keys

    base_minute = args.s3_prefix or 'global_forex/minute_aggs_v1'
    base_quotes = args.s3_prefix or 'global_forex/quotes_v1'

    # If user passed a custom prefix, allow overriding which subfolder to use
    if args.s3_prefix and 'minute' not in args.s3_prefix:
        base_minute = args.s3_prefix.rstrip('/') + '/minute_aggs_v1'
        base_quotes = args.s3_prefix.rstrip('/') + '/quotes_v1'

    keys = []
    keys += build_daily_keys(start, end, base_minute)
    keys += build_daily_keys(start, end, base_quotes)

    # Optionally filter keys by symbol substring if keys are per-symbol (not the case for Polygon daily files)
    if args.symbol and False:
        keys = [k for k in keys if args.symbol in k]

    if not keys:
        logger.error('No S3 keys generated for download - nothing to do')
        return

    # create temp dir
    if args.temp_dir:
        tmp = Path(args.temp_dir)
        tmp.mkdir(parents=True, exist_ok=True)
    else:
        tmp = Path(tempfile.mkdtemp(prefix='polygon_raw_'))
    logger.info('Using temp dir %s', tmp)

    try:
        local_files = download_s3_objects(s3, args.s3_bucket, keys, tmp)
        # read and classify raw files into OHLCV vs quotes
        ohlcv_dfs = []
        quote_dfs = []
        for f in local_files:
            try:
                df = read_raw_file(f)
                typ = classify_dataframe(df)
                if typ == 'ohlcv':
                    ohlcv_dfs.append(df)
                elif typ == 'quotes':
                    quote_dfs.append(df)
                else:
                    logger.warning('Unknown file type for %s - skipping', f)
            except Exception as e:
                logger.warning('Skipping file %s: %s', f, e)

        if not ohlcv_dfs and not quote_dfs:
            logger.error('No usable raw files could be read')
            return

        # Concatenate and filter for the requested ticker/symbol early to avoid extra processing
        df_ohlcv_all = pd.concat(ohlcv_dfs, ignore_index=True) if ohlcv_dfs else pd.DataFrame()
        df_quote_all = pd.concat(quote_dfs, ignore_index=True) if quote_dfs else pd.DataFrame()

        # Accept either 'ticker' or 'symbol' as the instrument column name
        if not df_ohlcv_all.empty:
            instrument_col = 'ticker' if 'ticker' in df_ohlcv_all.columns else ('symbol' if 'symbol' in df_ohlcv_all.columns else None)
            if instrument_col:
                df_ohlcv_all = df_ohlcv_all[df_ohlcv_all[instrument_col] == args.symbol]
            df_ohlcv_all['timestamp'] = pd.to_datetime(df_ohlcv_all['timestamp'])
            df_ohlcv_all = df_ohlcv_all[(df_ohlcv_all['timestamp'] >= pd.to_datetime(args.start)) & (df_ohlcv_all['timestamp'] <= pd.to_datetime(args.end))]

        if not df_quote_all.empty:
            instrument_col_q = 'ticker' if 'ticker' in df_quote_all.columns else ('symbol' if 'symbol' in df_quote_all.columns else None)
            if instrument_col_q:
                df_quote_all = df_quote_all[df_quote_all[instrument_col_q] == args.symbol]
            df_quote_all['timestamp'] = pd.to_datetime(df_quote_all['timestamp'])
            df_quote_all = df_quote_all[(df_quote_all['timestamp'] >= pd.to_datetime(args.start)) & (df_quote_all['timestamp'] <= pd.to_datetime(args.end))]

        # If we have quote minute-level files, aggregate them to minute bars
        if not df_quote_all.empty:
            # If quote files are already aggregated per minute (have window_start), aggregate_quotes_to_minute will handle grouping
            df_min_quote = aggregate_quotes_to_minute(df_quote_all)
            logger.info('Aggregated quote minutes: %d', len(df_min_quote))
        else:
            df_min_quote = pd.DataFrame()

        # If we have OHLCV minute files, use them directly (they may already be minute-level)
        if not df_ohlcv_all.empty:
            # Normalize columns to expected names
            df_tmp = df_ohlcv_all.rename(columns={
                'window_start': 'timestamp'
            })
            # Drop duplicated column labels (keep first occurrence) to avoid pandas "label not unique" errors
            if df_tmp.columns.duplicated().any():
                df_tmp = df_tmp.loc[:, ~df_tmp.columns.duplicated()]
            # Select core OHLCV columns if present
            missing = [c for c in ['timestamp','open','high','low','close','volume'] if c not in df_tmp.columns]
            if missing:
                raise ValueError(f'Missing required OHLCV columns after normalization: {missing}')
            df_min_ohlcv = df_tmp[['timestamp','open','high','low','close','volume']].sort_values('timestamp').reset_index(drop=True)
        else:
            # Build OHLCV from quote minutes if available
            if not df_min_quote.empty:
                df_min_ohlcv = resample_ohlcv_from_quotes(df_min_quote)
            else:
                df_min_ohlcv = pd.DataFrame()

        out_dir = Path(args.out)
        for tf in args.timeframes:
            df_tf = compute_timeframe_features(df_min_ohlcv, df_min_quote, tf)
            save_path = out_dir / args.symbol / f"{args.symbol}_{tf}.parquet"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_tf.to_parquet(save_path, index=False)
            logger.info('Saved timeframe parquet: %s', save_path)

    finally:
        if args.delete_local:
            try:
                shutil.rmtree(tmp)
                logger.info('Deleted local temp dir %s', tmp)
            except Exception:
                logger.warning('Failed to delete temp dir %s', tmp)


if __name__ == '__main__':
    main()
