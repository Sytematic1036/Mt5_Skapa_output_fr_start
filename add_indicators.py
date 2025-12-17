"""
Predictive Indicators Generator
===============================
Adds the following columns to all instrument files:
- TrendStrength
- Momentum
- VolRegime
- RegimeFilter (based on VIX z-score)
- TimeDecay
- TotalScore
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# =============================================================================
# CONFIG
# =============================================================================
INPUT_DIR = r"C:\Users\macka\Koder\Mt5_Skapa_output_fr_start\Download\Instruments"
OUTPUT_DIR = r"C:\Users\macka\Koder\Mt5_Skapa_output_fr_start\Download\Instruments_with_indicators"

# Column names (lowercase as in your files)
COL_TIME = "time"
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"

# Lookback periods
ATR_PERIOD = 14
EMA_PERIOD = 50
SLOPE_LOOKBACK = 10
MOM_LOOKBACK = 10
VOL_ROLL = 252  # 1 year for vol regime
VIX_ZSCORE_WINDOW = 252  # Rolling window for VIX z-score

# Score scaling
TREND_SCALE = 20.0
MOM_SCALE = 15.0
VOL_SCALE = 10.0
REGIME_SCALE = 8.0
TIME_DECAY_SCALE = 10.0

# Clipping
CLIP_ATR_UNITS = 3.0
CLIP_Z = 3.0


# =============================================================================
# VIX DOWNLOAD & REGIME FILTER
# =============================================================================
def download_vix_20_years():
    """Download 20 years of VIX data from Yahoo Finance"""
    print("=" * 70)
    print("DOWNLOADING VIX DATA (20 years)")
    print("=" * 70)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 20)

    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    vix = yf.download("^VIX", start=start_date.strftime('%Y-%m-%d'),
                      end=end_date.strftime('%Y-%m-%d'), progress=False)

    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    if vix.index.tz:
        vix.index = vix.index.tz_localize(None)

    vix = vix.reset_index()
    vix.columns = [c.lower() for c in vix.columns]
    vix = vix.rename(columns={'date': 'time'})

    print(f"Downloaded: {len(vix)} rows")
    print(f"Date range: {vix['time'].min()} to {vix['time'].max()}")

    return vix


def calculate_regime_filter(vix_df):
    """
    Calculate RegimeFilter from VIX using the professional method:
    1. Rolling z-score (252 days)
    2. Clip to [-3, +3]
    3. Invert sign (high VIX = negative)
    4. Normalize to [-1, +1]
    5. Scale to [-8, +8]
    """
    print("\n" + "=" * 70)
    print("CALCULATING REGIME FILTER")
    print("=" * 70)

    df = vix_df.copy()
    df = df.sort_values('time').reset_index(drop=True)

    vix_close = df['close'].astype(float)

    # Step 1: Rolling z-score (252 days)
    rolling_mean = vix_close.rolling(window=VIX_ZSCORE_WINDOW, min_periods=50).mean()
    rolling_std = vix_close.rolling(window=VIX_ZSCORE_WINDOW, min_periods=50).std(ddof=0)
    z_vix = (vix_close - rolling_mean) / rolling_std.replace(0, np.nan)

    # Step 2: Clip to [-3, +3]
    z_vix_clipped = z_vix.clip(lower=-CLIP_Z, upper=CLIP_Z)

    # Step 3: Invert sign (high VIX = bad = negative)
    regime_raw = -z_vix_clipped

    # Step 4: Normalize to [-1, +1]
    regime_norm = regime_raw / CLIP_Z

    # Step 5: Scale to [-8, +8]
    regime_filter = regime_norm * REGIME_SCALE

    df['vix_z'] = z_vix
    df['regime_filter'] = regime_filter

    # Stats
    valid = regime_filter.dropna()
    print(f"RegimeFilter stats:")
    print(f"  Min: {valid.min():.2f}")
    print(f"  Max: {valid.max():.2f}")
    print(f"  Mean: {valid.mean():.2f}")
    print(f"  Std: {valid.std():.2f}")

    return df[['time', 'regime_filter']]


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================
def wilder_atr(high, low, close, period=14):
    """Wilder's ATR using exponential smoothing"""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    return atr


def ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rolling_zscore(series, window):
    """Rolling z-score"""
    mu = series.rolling(window=window, min_periods=max(20, window//5)).mean()
    sd = series.rolling(window=window, min_periods=max(20, window//5)).std(ddof=0)
    z = (series - mu) / sd.replace(0, np.nan)
    return z


def compute_time_decay(trend_dir, hold_days=10):
    """
    Time-decay: penalty when same trend direction persists too long
    """
    run_len = np.zeros(len(trend_dir), dtype=int)
    prev = 0
    streak = 0

    for i, v in enumerate(trend_dir.fillna(0).astype(int).values):
        if v != 0 and v == prev:
            streak += 1
        elif v != 0:
            streak = 1
        else:
            streak = 0
        run_len[i] = streak
        prev = v

    run_len = pd.Series(run_len, index=trend_dir.index)
    decay_strength = (run_len - hold_days) / float(hold_days)
    decay_strength = decay_strength.clip(lower=0.0, upper=1.0)
    time_decay = -TIME_DECAY_SCALE * decay_strength

    return time_decay


def add_indicators(df, regime_df):
    """
    Add all predictive indicator columns to a dataframe
    """
    out = df.copy()
    out[COL_TIME] = pd.to_datetime(out[COL_TIME])
    out = out.sort_values(COL_TIME).reset_index(drop=True)

    close = out[COL_CLOSE].astype(float)
    high = out[COL_HIGH].astype(float)
    low = out[COL_LOW].astype(float)

    # --- ATR ---
    atr = wilder_atr(high, low, close, period=ATR_PERIOD)
    out["ATR14"] = atr

    # --- EMA & TrendStrength ---
    e = ema(close, EMA_PERIOD)
    out[f"EMA{EMA_PERIOD}"] = e

    slope = e - e.shift(SLOPE_LOOKBACK)
    trend_atr_units = slope / atr
    trend_atr_units = trend_atr_units.clip(lower=-CLIP_ATR_UNITS, upper=CLIP_ATR_UNITS)
    trend_strength = trend_atr_units * TREND_SCALE
    out["TrendStrength"] = trend_strength

    # Trend direction for time-decay
    trend_dir = np.sign(trend_atr_units).replace(0, 0)
    out["TrendDir"] = trend_dir

    # --- Momentum ---
    mom_raw = close - close.shift(MOM_LOOKBACK)
    mom_atr_units = mom_raw / atr
    mom_atr_units = mom_atr_units.clip(lower=-CLIP_ATR_UNITS, upper=CLIP_ATR_UNITS)
    momentum = mom_atr_units * MOM_SCALE
    out["Momentum"] = momentum

    # --- VolRegime ---
    z_atr = rolling_zscore(atr, VOL_ROLL)
    z_atr = z_atr.clip(lower=-CLIP_Z, upper=CLIP_Z)
    out["ATR_Z"] = z_atr

    # Vol score: highest near z=0, lower at extremes
    vol_score = (1.0 - (z_atr.abs() / CLIP_Z)) * VOL_SCALE
    vol_score = vol_score.fillna(0.0)
    out["VolRegime"] = vol_score

    # --- RegimeFilter (merge from VIX) ---
    regime_df_copy = regime_df.copy()
    regime_df_copy['time'] = pd.to_datetime(regime_df_copy['time'])

    merged = out[[COL_TIME]].merge(regime_df_copy, left_on=COL_TIME, right_on='time', how='left')
    reg = merged['regime_filter'].astype(float)
    reg = reg.ffill().fillna(0.0)
    out["RegimeFilter"] = reg.values

    # --- TimeDecay ---
    time_decay = compute_time_decay(out["TrendDir"], hold_days=MOM_LOOKBACK)
    out["TimeDecay"] = time_decay

    # --- TotalScore ---
    out["TotalScore"] = (
        out["TrendStrength"] +
        out["Momentum"] +
        out["VolRegime"] +
        out["RegimeFilter"] +
        out["TimeDecay"]
    )

    return out


# =============================================================================
# MAIN PROCESSING
# =============================================================================
def process_all_instruments(input_dir, output_dir, regime_df):
    """Process all instrument files and add indicators"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    input_path = Path(input_dir)
    files = sorted([p for p in input_path.glob("*.xlsx") if not p.name.startswith("~$")])

    # Exclude VIX if it exists (we're using yfinance VIX)
    files = [f for f in files if 'VIX' not in f.name.upper()]

    print(f"\n" + "=" * 70)
    print(f"PROCESSING {len(files)} INSTRUMENTS")
    print("=" * 70)

    success = 0
    failed = 0

    for i, f in enumerate(files, 1):
        try:
            df = pd.read_excel(f)
            scored = add_indicators(df, regime_df)

            out_file = Path(output_dir) / f.name
            scored.to_excel(out_file, index=False)

            print(f"[{i:3}/{len(files)}] OK: {f.name} ({len(scored)} rows)")
            success += 1

        except Exception as e:
            print(f"[{i:3}/{len(files)}] FAILED: {f.name} -> {e}")
            failed += 1

    return success, failed


def save_vix_as_instrument(vix_df, regime_df, output_dir):
    """Save VIX as an instrument file with indicators"""

    # Prepare VIX dataframe in instrument format
    vix = vix_df.copy()
    vix = vix.rename(columns={
        'time': COL_TIME,
        'open': COL_OPEN,
        'high': COL_HIGH,
        'low': COL_LOW,
        'close': COL_CLOSE
    })

    # Add missing columns
    if 'tick_volume' not in vix.columns:
        vix['tick_volume'] = 0
    if 'spread' not in vix.columns:
        vix['spread'] = 0

    vix = vix[[COL_TIME, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, 'tick_volume', 'spread']]

    # Add indicators
    scored = add_indicators(vix, regime_df)

    out_file = Path(output_dir) / "VIX_Index.xlsx"
    scored.to_excel(out_file, index=False)
    print(f"\nVIX saved: {out_file} ({len(scored)} rows)")


def main():
    print("\n" + "=" * 70)
    print("PREDICTIVE INDICATORS GENERATOR")
    print("=" * 70)

    # Step 1: Download VIX
    vix_df = download_vix_20_years()

    # Step 2: Calculate RegimeFilter from VIX
    regime_df = calculate_regime_filter(vix_df)

    # Step 3: Process all instruments
    success, failed = process_all_instruments(INPUT_DIR, OUTPUT_DIR, regime_df)

    # Step 4: Save VIX as instrument
    save_vix_as_instrument(vix_df, regime_df, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {success} instruments")
    print(f"Failed: {failed} instruments")
    print(f"VIX: saved with indicators")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Show sample of indicator columns
    print("\nNew columns added:")
    print("  - ATR14")
    print("  - EMA50")
    print("  - TrendStrength")
    print("  - TrendDir")
    print("  - Momentum")
    print("  - ATR_Z")
    print("  - VolRegime")
    print("  - RegimeFilter (from VIX z-score)")
    print("  - TimeDecay")
    print("  - TotalScore")


if __name__ == "__main__":
    main()
