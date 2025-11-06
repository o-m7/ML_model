#!/usr/bin/env python3
"""
Live Market Features Sync
Continuously fetches latest Polygon data and calculates comprehensive technical indicators.
Updates Supabase features table every 3 seconds.
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

try:
    import talib
    from supabase import create_client
except ImportError:
    print("ERROR: Missing dependencies. Install with:")
    print("pip install TA-Lib pandas numpy supabase python-dotenv requests")
    sys.exit(1)

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ifetofkhyblyijghuwzs.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not all([POLYGON_API_KEY, SUPABASE_SERVICE_KEY]):
    print("ERROR: Missing POLYGON_API_KEY or SUPABASE_SERVICE_ROLE_KEY")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Symbol configurations
CONFIGS = [
    {'symbol': 'XAUUSD', 'polygon_symbol': 'C:XAUUSD', 'timeframes': ['15T', '30T', '1H', '4H']},
    {'symbol': 'XAGUSD', 'polygon_symbol': 'C:XAGUSD', 'timeframes': ['15T', '30T', '1H', '4H']},
]

TF_MAP = {
    '15T': (15, 'minute'),
    '30T': (30, 'minute'),
    '1H': (1, 'hour'),
    '4H': (4, 'hour'),
}


def fetch_polygon_data(polygon_symbol: str, multiplier: int, timespan: str, limit: int = 500):
    """Fetch OHLCV data from Polygon"""
    to_date = datetime.now()
    from_date = to_date - timedelta(days=60)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/{multiplier}/{timespan}/{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': limit,
        'apiKey': POLYGON_API_KEY
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'results' not in data or not data['results']:
        return None
    
    df = pd.DataFrame(data['results'])
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators matching the parquet structure"""
    
    if len(df) < 200:
        print(f"Warning: Not enough data ({len(df)} bars), need at least 200")
        return None
    
    o, h, l, c, v = df['open'].values, df['high'].values, df['low'].values, df['close'].values, df['volume'].values
    
    # Create feature dict for the latest candle
    latest = df.iloc[-1].copy()
    features = {
        'timestamp': latest['timestamp'],
        'open': latest['open'],
        'high': latest['high'],
        'low': latest['low'],
        'close': latest['close'],
        'volume': latest['volume'],
    }
    
    # ATR and range
    features['atr14'] = talib.ATR(h, l, c, timeperiod=14)[-1]
    features['trange'] = talib.TRANGE(h, l, c)[-1]
    features['natr'] = talib.NATR(h, l, c, timeperiod=14)[-1]
    features['atr'] = features['atr14']
    
    # Volume percentile
    vol_rank = pd.Series(v).rank(pct=True)
    features['vol_percentile'] = vol_rank.iloc[-1]
    features['vol_percentile_4H'] = vol_rank.iloc[-1]  # Placeholder
    
    # Bollinger Bands
    bb_up, bb_mid, bb_lo = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    features['bb_mid_20'] = bb_mid[-1]
    features['bb_up_20'] = bb_up[-1]
    features['bb_lo_20'] = bb_lo[-1]
    features['bb_upper_20'] = bb_up[-1]
    features['bb_lower_20'] = bb_lo[-1]
    features['bb_width_20'] = (bb_up[-1] - bb_lo[-1]) / bb_mid[-1] if bb_mid[-1] != 0 else 0
    features['bb_bw_20'] = features['bb_width_20']
    features['bb_bbp_20'] = (c[-1] - bb_lo[-1]) / (bb_up[-1] - bb_lo[-1]) if (bb_up[-1] - bb_lo[-1]) != 0 else 0
    features['bb_pctb_20'] = features['bb_bbp_20']
    features['bb_bw_20_4H'] = features['bb_width_20']  # Placeholder
    
    # BB 50
    bb_up_50, bb_mid_50, bb_lo_50 = talib.BBANDS(c, timeperiod=50, nbdevup=2, nbdevdn=2, matype=0)
    features['bb_upper_50'] = bb_up_50[-1]
    features['bb_mid_50'] = bb_mid_50[-1]
    features['bb_lower_50'] = bb_lo_50[-1]
    features['bb_width_50'] = (bb_up_50[-1] - bb_lo_50[-1]) / bb_mid_50[-1] if bb_mid_50[-1] != 0 else 0
    features['bb_pctb_50'] = (c[-1] - bb_lo_50[-1]) / (bb_up_50[-1] - bb_lo_50[-1]) if (bb_up_50[-1] - bb_lo_50[-1]) != 0 else 0
    
    # Keltner Channels (approximation using ATR)
    kelt_mid = talib.EMA(c, timeperiod=20)[-1]
    atr = talib.ATR(h, l, c, timeperiod=20)[-1]
    features['kelt_mid_20'] = kelt_mid
    features['kelt_up_20'] = kelt_mid + (2 * atr)
    features['kelt_lo_20'] = kelt_mid - (2 * atr)
    features['kelt_bw_20'] = (4 * atr) / kelt_mid if kelt_mid != 0 else 0
    features['kelt_bw_20_4H'] = features['kelt_bw_20']  # Placeholder
    
    # Squeeze
    features['squeeze_on'] = 1 if features['bb_width_20'] < features['kelt_bw_20'] else 0
    features['squeeze_on_4H'] = features['squeeze_on']  # Placeholder
    
    # EMAs
    features['ema20'] = talib.EMA(c, timeperiod=20)[-1]
    features['ema50'] = talib.EMA(c, timeperiod=50)[-1]
    features['ema200'] = talib.EMA(c, timeperiod=200)[-1]
    features['ema_5'] = talib.EMA(c, timeperiod=5)[-1]
    features['ema_10'] = talib.EMA(c, timeperiod=10)[-1]
    features['ema_20'] = features['ema20']
    features['ema_50'] = features['ema50']
    features['ema_100'] = talib.EMA(c, timeperiod=100)[-1]
    features['ema_200'] = features['ema200']
    
    # EMA slopes
    ema20_series = talib.EMA(c, timeperiod=20)
    ema50_series = talib.EMA(c, timeperiod=50)
    ema200_series = talib.EMA(c, timeperiod=200)
    features['ema20_slope'] = (ema20_series[-1] - ema20_series[-5]) / ema20_series[-5] if len(ema20_series) >= 5 else 0
    features['ema50_slope'] = (ema50_series[-1] - ema50_series[-5]) / ema50_series[-5] if len(ema50_series) >= 5 else 0
    features['ema200_slope'] = (ema200_series[-1] - ema200_series[-5]) / ema200_series[-5] if len(ema200_series) >= 5 else 0
    features['ema20_slope_4H'] = features['ema20_slope']
    features['ema50_slope_4H'] = features['ema50_slope']
    features['ema200_slope_4H'] = features['ema200_slope']
    
    features['ema20_4H'] = features['ema20']
    features['ema50_4H'] = features['ema50']
    features['ema200_4H'] = features['ema200']
    
    # SMAs
    features['sma_5'] = talib.SMA(c, timeperiod=5)[-1]
    features['sma_10'] = talib.SMA(c, timeperiod=10)[-1]
    features['sma_20'] = talib.SMA(c, timeperiod=20)[-1]
    features['sma_50'] = talib.SMA(c, timeperiod=50)[-1]
    features['sma_100'] = talib.SMA(c, timeperiod=100)[-1]
    features['sma_200'] = talib.SMA(c, timeperiod=200)[-1]
    
    # WMAs
    features['wma_5'] = talib.WMA(c, timeperiod=5)[-1]
    features['wma_10'] = talib.WMA(c, timeperiod=10)[-1]
    features['wma_20'] = talib.WMA(c, timeperiod=20)[-1]
    features['wma_50'] = talib.WMA(c, timeperiod=50)[-1]
    features['wma_100'] = talib.WMA(c, timeperiod=100)[-1]
    features['wma_200'] = talib.WMA(c, timeperiod=200)[-1]
    
    # Other MAs
    features['dema_10'] = talib.DEMA(c, timeperiod=10)[-1]
    features['tema_10'] = talib.TEMA(c, timeperiod=10)[-1]
    features['trima_10'] = talib.TRIMA(c, timeperiod=10)[-1]
    features['dema_20'] = talib.DEMA(c, timeperiod=20)[-1]
    features['tema_20'] = talib.TEMA(c, timeperiod=20)[-1]
    features['trima_20'] = talib.TRIMA(c, timeperiod=20)[-1]
    features['dema_30'] = talib.DEMA(c, timeperiod=30)[-1]
    features['tema_30'] = talib.TEMA(c, timeperiod=30)[-1]
    features['trima_30'] = talib.TRIMA(c, timeperiod=30)[-1]
    features['kama_30'] = talib.KAMA(c, timeperiod=30)[-1]
    
    mama, fama = talib.MAMA(c)
    features['mama'] = mama[-1]
    features['fama'] = fama[-1]
    
    features['midpoint'] = talib.MIDPOINT(c, timeperiod=14)[-1]
    features['midprice'] = talib.MIDPRICE(h, l, timeperiod=14)[-1]
    features['sar'] = talib.SAR(h, l)[-1]
    features['sarext'] = talib.SAREXT(h, l)[-1]
    features['t3'] = talib.T3(c, timeperiod=5)[-1]
    features['ht_trendline'] = talib.HT_TRENDLINE(c)[-1]
    
    # Trend strength (simplified)
    features['trend_strength'] = abs(features['ema20_slope']) * 100
    features['trend_strength_4H'] = features['trend_strength']
    
    # Pullback (placeholder)
    features['pullback_depth'] = 0.0
    features['pullback_time'] = 0
    
    # ADX and Aroon
    features['adx14'] = talib.ADX(h, l, c, timeperiod=14)[-1]
    features['adx'] = features['adx14']
    features['adxr'] = talib.ADXR(h, l, c, timeperiod=14)[-1]
    features['dx'] = talib.DX(h, l, c, timeperiod=14)[-1]
    
    aroon_down, aroon_up = talib.AROON(h, l, timeperiod=25)
    features['aroon_up'] = aroon_up[-1]
    features['aroon_dn'] = aroon_down[-1]
    features['aroon_down'] = aroon_down[-1]
    features['aroon_osc'] = talib.AROONOSC(h, l, timeperiod=25)[-1]
    
    # Donchian
    features['donch_up20'] = talib.MAX(h, timeperiod=20)[-1]
    features['donch_lo20'] = talib.MIN(l, timeperiod=20)[-1]
    features['donch_mid20'] = (features['donch_up20'] + features['donch_lo20']) / 2
    features['donch_up20_4H'] = features['donch_up20']
    features['donch_lo20_4H'] = features['donch_lo20']
    
    # RSI
    features['rsi14'] = talib.RSI(c, timeperiod=14)[-1]
    features['rsi_7'] = talib.RSI(c, timeperiod=7)[-1]
    features['rsi_14'] = features['rsi14']
    features['rsi_21'] = talib.RSI(c, timeperiod=21)[-1]
    features['rsi14_4H'] = features['rsi14']
    
    # MACD
    macd, macds, macdh = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    features['macd'] = macd[-1]
    features['macds'] = macds[-1]
    features['macdh'] = macdh[-1]
    features['macd_signal'] = macds[-1]
    features['macd_hist'] = macdh[-1]
    features['macd_4H'] = macd[-1]
    features['macds_4H'] = macds[-1]
    features['macdh_4H'] = macdh[-1]
    
    macdext, macdext_signal, macdext_hist = talib.MACDEXT(c)
    features['macdext'] = macdext[-1]
    features['macdext_signal'] = macdext_signal[-1]
    features['macdext_hist'] = macdext_hist[-1]
    
    macdfix, macdfix_signal, macdfix_hist = talib.MACDFIX(c)
    features['macdfix'] = macdfix[-1]
    features['macdfix_signal'] = macdfix_signal[-1]
    features['macdfix_hist'] = macdfix_hist[-1]
    
    features['apo'] = talib.APO(c)[-1]
    features['ppo'] = talib.PPO(c)[-1]
    
    # Stochastic
    slowk, slowd = talib.STOCH(h, l, c)
    features['stoch_k'] = slowk[-1]
    features['stoch_d'] = slowd[-1]
    features['slowk'] = slowk[-1]
    features['slowd'] = slowd[-1]
    
    fastk, fastd = talib.STOCHF(h, l, c)
    features['fastk'] = fastk[-1]
    features['fastd'] = fastd[-1]
    
    fastk_rsi, fastd_rsi = talib.STOCHRSI(c)
    features['fastk_rsi'] = fastk_rsi[-1]
    features['fastd_rsi'] = fastd_rsi[-1]
    
    # Other momentum
    features['bop'] = talib.BOP(o, h, l, c)[-1]
    features['cci'] = talib.CCI(h, l, c, timeperiod=14)[-1]
    features['cmo'] = talib.CMO(c, timeperiod=14)[-1]
    features['mfi'] = talib.MFI(h, l, c, v, timeperiod=14)[-1]
    features['minus_di'] = talib.MINUS_DI(h, l, c, timeperiod=14)[-1]
    features['minus_dm'] = talib.MINUS_DM(h, l, timeperiod=14)[-1]
    features['plus_di'] = talib.PLUS_DI(h, l, c, timeperiod=14)[-1]
    features['plus_dm'] = talib.PLUS_DM(h, l, timeperiod=14)[-1]
    features['mom'] = talib.MOM(c, timeperiod=10)[-1]
    features['roc'] = talib.ROC(c, timeperiod=10)[-1]
    features['rocp'] = talib.ROCP(c, timeperiod=10)[-1]
    features['rocr'] = talib.ROCR(c, timeperiod=10)[-1]
    features['rocr100'] = talib.ROCR100(c, timeperiod=10)[-1]
    features['trix'] = talib.TRIX(c, timeperiod=30)[-1]
    features['ultosc'] = talib.ULTOSC(h, l, c)[-1]
    features['willr'] = talib.WILLR(h, l, c, timeperiod=14)[-1]
    
    # Volume
    features['obv'] = talib.OBV(c, v)[-1]
    features['ad'] = talib.AD(h, l, c, v)[-1]
    features['adosc'] = talib.ADOSC(h, l, c, v)[-1]
    
    # Price
    features['avgprice'] = talib.AVGPRICE(o, h, l, c)[-1]
    features['medprice'] = talib.MEDPRICE(h, l)[-1]
    features['typprice'] = talib.TYPPRICE(h, l, c)[-1]
    features['wclprice'] = talib.WCLPRICE(h, l, c)[-1]
    
    # Hilbert Transform
    features['ht_dcperiod'] = talib.HT_DCPERIOD(c)[-1]
    features['ht_dcphase'] = talib.HT_DCPHASE(c)[-1]
    inphase, quadrature = talib.HT_PHASOR(c)
    features['ht_inphase'] = inphase[-1]
    features['ht_quadrature'] = quadrature[-1]
    sine, leadsine = talib.HT_SINE(c)
    features['ht_sine'] = sine[-1]
    features['ht_leadsine'] = leadsine[-1]
    features['ht_trendmode'] = talib.HT_TRENDMODE(c)[-1]
    
    # Candle patterns (all TA-Lib patterns)
    features['cdl2crows'] = talib.CDL2CROWS(o, h, l, c)[-1]
    features['cdl3blackcrows'] = talib.CDL3BLACKCROWS(o, h, l, c)[-1]
    features['cdl3inside'] = talib.CDL3INSIDE(o, h, l, c)[-1]
    features['cdl3linestrike'] = talib.CDL3LINESTRIKE(o, h, l, c)[-1]
    features['cdl3outside'] = talib.CDL3OUTSIDE(o, h, l, c)[-1]
    features['cdl3starsinsouth'] = talib.CDL3STARSINSOUTH(o, h, l, c)[-1]
    features['cdl3whitesoldiers'] = talib.CDL3WHITESOLDIERS(o, h, l, c)[-1]
    features['cdlabandonedbaby'] = talib.CDLABANDONEDBABY(o, h, l, c)[-1]
    features['cdladvanceblock'] = talib.CDLADVANCEBLOCK(o, h, l, c)[-1]
    features['cdlbelthold'] = talib.CDLBELTHOLD(o, h, l, c)[-1]
    features['cdlbreakaway'] = talib.CDLBREAKAWAY(o, h, l, c)[-1]
    features['cdlclosingmarubozu'] = talib.CDLCLOSINGMARUBOZU(o, h, l, c)[-1]
    features['cdlconcealbabyswall'] = talib.CDLCONCEALBABYSWALL(o, h, l, c)[-1]
    features['cdlcounterattack'] = talib.CDLCOUNTERATTACK(o, h, l, c)[-1]
    features['cdldarkcloudcover'] = talib.CDLDARKCLOUDCOVER(o, h, l, c)[-1]
    features['cdldoji'] = talib.CDLDOJI(o, h, l, c)[-1]
    features['cdldojistar'] = talib.CDLDOJISTAR(o, h, l, c)[-1]
    features['cdldragonflydoji'] = talib.CDLDRAGONFLYDOJI(o, h, l, c)[-1]
    features['cdlengulfing'] = talib.CDLENGULFING(o, h, l, c)[-1]
    features['cdleveningdojistar'] = talib.CDLEVENINGDOJISTAR(o, h, l, c)[-1]
    features['cdleveningstar'] = talib.CDLEVENINGSTAR(o, h, l, c)[-1]
    features['cdlgapsidesidewhite'] = talib.CDLGAPSIDESIDEWHITE(o, h, l, c)[-1]
    features['cdlgravestonedoji'] = talib.CDLGRAVESTONEDOJI(o, h, l, c)[-1]
    features['cdlhammer'] = talib.CDLHAMMER(o, h, l, c)[-1]
    features['cdlhangingman'] = talib.CDLHANGINGMAN(o, h, l, c)[-1]
    features['cdlharami'] = talib.CDLHARAMI(o, h, l, c)[-1]
    features['cdlharamicross'] = talib.CDLHARAMICROSS(o, h, l, c)[-1]
    features['cdlhighwave'] = talib.CDLHIGHWAVE(o, h, l, c)[-1]
    features['cdlhikkake'] = talib.CDLHIKKAKE(o, h, l, c)[-1]
    features['cdlhikkakemod'] = talib.CDLHIKKAKEMOD(o, h, l, c)[-1]
    features['cdlhomingpigeon'] = talib.CDLHOMINGPIGEON(o, h, l, c)[-1]
    features['cdlidentical3crows'] = talib.CDLIDENTICAL3CROWS(o, h, l, c)[-1]
    features['cdlinneck'] = talib.CDLINNECK(o, h, l, c)[-1]
    features['cdlinvertedhammer'] = talib.CDLINVERTEDHAMMER(o, h, l, c)[-1]
    features['cdlkicking'] = talib.CDLKICKING(o, h, l, c)[-1]
    features['cdlkickingbylength'] = talib.CDLKICKINGBYLENGTH(o, h, l, c)[-1]
    features['cdlladderbottom'] = talib.CDLLADDERBOTTOM(o, h, l, c)[-1]
    features['cdllongleggeddoji'] = talib.CDLLONGLEGGEDDOJI(o, h, l, c)[-1]
    features['cdllongline'] = talib.CDLLONGLINE(o, h, l, c)[-1]
    features['cdlmarubozu'] = talib.CDLMARUBOZU(o, h, l, c)[-1]
    features['cdlmatchinglow'] = talib.CDLMATCHINGLOW(o, h, l, c)[-1]
    features['cdlmathold'] = talib.CDLMATHOLD(o, h, l, c)[-1]
    features['cdlmorningdojistar'] = talib.CDLMORNINGDOJISTAR(o, h, l, c)[-1]
    features['cdlmorningstar'] = talib.CDLMORNINGSTAR(o, h, l, c)[-1]
    features['cdlonneck'] = talib.CDLONNECK(o, h, l, c)[-1]
    features['cdlpiercing'] = talib.CDLPIERCING(o, h, l, c)[-1]
    features['cdlrickshawman'] = talib.CDLRICKSHAWMAN(o, h, l, c)[-1]
    features['cdlrisefall3methods'] = talib.CDLRISEFALL3METHODS(o, h, l, c)[-1]
    features['cdlseparatinglines'] = talib.CDLSEPARATINGLINES(o, h, l, c)[-1]
    features['cdlshootingstar'] = talib.CDLSHOOTINGSTAR(o, h, l, c)[-1]
    features['cdlshortline'] = talib.CDLSHORTLINE(o, h, l, c)[-1]
    features['cdlspinningtop'] = talib.CDLSPINNINGTOP(o, h, l, c)[-1]
    features['cdlstalledpattern'] = talib.CDLSTALLEDPATTERN(o, h, l, c)[-1]
    features['cdlsticksandwich'] = talib.CDLSTICKSANDWICH(o, h, l, c)[-1]
    features['cdltakuri'] = talib.CDLTAKURI(o, h, l, c)[-1]
    features['cdltasukigap'] = talib.CDLTASUKIGAP(o, h, l, c)[-1]
    features['cdlthrusting'] = talib.CDLTHRUSTING(o, h, l, c)[-1]
    features['cdltristar'] = talib.CDLTRISTAR(o, h, l, c)[-1]
    features['cdlunique3river'] = talib.CDLUNIQUE3RIVER(o, h, l, c)[-1]
    features['cdlupsidegap2crows'] = talib.CDLUPSIDEGAP2CROWS(o, h, l, c)[-1]
    features['cdlxsidegap3methods'] = talib.CDLXSIDEGAP3METHODS(o, h, l, c)[-1]
    
    # Statistics
    features['beta'] = talib.BETA(h, l, timeperiod=5)[-1]
    features['correl'] = talib.CORREL(h, l, timeperiod=30)[-1]
    features['linearreg'] = talib.LINEARREG(c, timeperiod=14)[-1]
    features['linearreg_angle'] = talib.LINEARREG_ANGLE(c, timeperiod=14)[-1]
    features['linearreg_intercept'] = talib.LINEARREG_INTERCEPT(c, timeperiod=14)[-1]
    features['linearreg_slope'] = talib.LINEARREG_SLOPE(c, timeperiod=14)[-1]
    features['stddev'] = talib.STDDEV(c, timeperiod=5)[-1]
    features['tsf'] = talib.TSF(c, timeperiod=14)[-1]
    features['var'] = talib.VAR(c, timeperiod=5)[-1]
    
    # Max/Min/Sum
    features['max_10'] = talib.MAX(c, timeperiod=10)[-1]
    features['min_10'] = talib.MIN(c, timeperiod=10)[-1]
    features['sum_10'] = talib.SUM(c, timeperiod=10)[-1]
    features['max_20'] = talib.MAX(c, timeperiod=20)[-1]
    features['min_20'] = talib.MIN(c, timeperiod=20)[-1]
    features['sum_20'] = talib.SUM(c, timeperiod=20)[-1]
    
    # Custom features (placeholders for now)
    body = abs(c[-1] - o[-1])
    upper_wick = h[-1] - max(o[-1], c[-1])
    lower_wick = min(o[-1], c[-1]) - l[-1]
    total_range = h[-1] - l[-1]
    
    features['upper_wick_ratio'] = upper_wick / total_range if total_range != 0 else 0
    features['lower_wick_ratio'] = lower_wick / total_range if total_range != 0 else 0
    features['wick_pressure'] = features['upper_wick_ratio'] - features['lower_wick_ratio']
    
    # Time features
    ts = latest['timestamp']
    features['minute_of_day'] = ts.hour * 60 + ts.minute
    features['dow'] = ts.dayofweek
    
    # Session placeholders
    features['session'] = 'US' if 13 <= ts.hour <= 20 else ('EU' if 7 <= ts.hour <= 16 else 'ASIA')
    features['session_pos'] = 0.5
    features['sess_asia'] = 1 if features['session'] == 'ASIA' else 0
    features['sess_eu'] = 1 if features['session'] == 'EU' else 0
    features['sess_us'] = 1 if features['session'] == 'US' else 0
    
    # Structure placeholders
    features['fvg_bull'] = 0
    features['fvg_bear'] = 0
    features['fvg_bull_sz'] = 0.0
    features['fvg_bear_sz'] = 0.0
    features['swing_high'] = 0
    features['swing_low'] = 0
    features['dist_nearest_sr0'] = 0.0
    features['bos_up'] = 0
    features['bos_dn'] = 0
    features['eq_high'] = 0
    features['eq_low'] = 0
    features['london_prev_high'] = h[-1]
    features['london_prev_low'] = l[-1]
    
    return features


def sync_symbol_timeframe(symbol: str, polygon_symbol: str, tf: str):
    """Sync one symbol/timeframe"""
    try:
        multiplier, timespan = TF_MAP[tf]
        df = fetch_polygon_data(polygon_symbol, multiplier, timespan)
        
        if df is None or len(df) < 200:
            print(f"  ⚠️ {symbol} {tf}: Insufficient data")
            return False
        
        features = calculate_all_features(df)
        if features is None:
            return False
        
        # Add metadata
        features['symbol'] = symbol
        features['tf'] = tf
        features['ts'] = features['timestamp'].isoformat()
        del features['timestamp']
        
        # Upsert to Supabase
        response = supabase.table('features').upsert(features, on_conflict='symbol,tf,ts').execute()
        
        print(f"  ✓ {symbol} {tf} @ {features['ts']}")
        return True
        
    except Exception as e:
        print(f"  ✗ {symbol} {tf}: {str(e)}")
        return False


def main():
    """Main sync loop"""
    print("Starting live features sync (every 3 seconds)...")
    print(f"Symbols: {', '.join([c['symbol'] for c in CONFIGS])}")
    print(f"Press Ctrl+C to stop\n")
    
    iteration = 0
    while True:
        iteration += 1
        print(f"[Iteration {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for config in CONFIGS:
            for tf in config['timeframes']:
                sync_symbol_timeframe(config['symbol'], config['polygon_symbol'], tf)
        
        print()
        time.sleep(3)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        sys.exit(0)
