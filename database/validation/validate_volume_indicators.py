#!/usr/bin/env python3
"""
Volume Indicators Validator
- Validates: obv_slope, volume_change_pct, volume_zone_oscillator, chaikin_money_flow, klinger, vwma, etc.
- Confirms directional change logic matches price/volume shifts
"""

import pandas as pd
import numpy as np
from database.validation.validation_utils import main_validator

def calculate_obv(close, volume):
    """Calculate On-Balance Volume (OBV) independently"""
    obv = np.zeros(len(close))
    
    # First value is same as first volume
    obv[0] = volume.iloc[0]
    
    # Calculate subsequent OBV values
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv[i] = obv[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv[i] = obv[i-1] - volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    
    # Calculate OBV slope (change over 3 periods)
    obv_series = pd.Series(obv, index=close.index)
    obv_slope = obv_series.diff(3) / 3
    
    return {
        'obv': obv_series,
        'obv_slope': obv_slope
    }

def calculate_money_flow_index(high, low, close, volume, length=14):
    """Calculate Money Flow Index (MFI) independently"""
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate money flow
    money_flow = typical_price * volume
    
    # Calculate directional money flow (using typical price changes)
    positive_flow = np.zeros(len(typical_price))
    negative_flow = np.zeros(len(typical_price))
    
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow[i] = money_flow.iloc[i]
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            negative_flow[i] = money_flow.iloc[i]
        else:
            # If prices are the same, split the flow proportionally
            positive_flow[i] = money_flow.iloc[i] / 2
            negative_flow[i] = money_flow.iloc[i] / 2
    
    # Convert to Series
    positive_flow = pd.Series(positive_flow, index=typical_price.index)
    negative_flow = pd.Series(negative_flow, index=typical_price.index)
    
    # Calculate positive and negative flow for the period
    positive_sum = positive_flow.rolling(window=length, min_periods=1).sum()
    negative_sum = negative_flow.rolling(window=length, min_periods=1).sum()
    
    # Calculate money ratio
    money_ratio = pd.Series(0, index=typical_price.index)
    valid_denom = negative_sum > 0
    money_ratio[valid_denom] = positive_sum[valid_denom] / negative_sum[valid_denom]
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi.fillna(50)  # Default to neutral 50 for NaN values

def calculate_vwma(close, volume, length=20):
    """Calculate Volume Weighted Moving Average (VWMA) independently"""
    weighted_price = close * volume
    
    # Calculate sum of the weights for the window
    volume_sum = volume.rolling(window=length, min_periods=1).sum()
    
    # Protect against division by zero
    vwma = pd.Series(0.0, index=close.index)
    valid_denom = volume_sum > 0
    
    # Calculate VWMA as sum(price * volume) / sum(volume)
    weighted_sum = weighted_price.rolling(window=length, min_periods=1).sum()
    vwma[valid_denom] = weighted_sum[valid_denom] / volume_sum[valid_denom]
    
    # Default to close price for invalid values
    vwma[~valid_denom] = close[~valid_denom]
    
    return vwma

def calculate_chaikin_money_flow(high, low, close, volume, length=20):
    """Calculate Chaikin Money Flow (CMF) independently"""
    # Money Flow Multiplier: ((Close - Low) - (High - Close)) / (High - Low)
    high_low_range = high - low
    
    # Avoid division by zero
    mfm = pd.Series(0.0, index=high.index)
    valid_range = high_low_range > 0
    
    # Only calculate for valid ranges
    mfm[valid_range] = ((close[valid_range] - low[valid_range]) - 
                         (high[valid_range] - close[valid_range])) / high_low_range[valid_range]
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # Chaikin Money Flow: Sum(MFV) / Sum(Volume) over period
    mfv_sum = mfv.rolling(window=length, min_periods=1).sum()
    vol_sum = volume.rolling(window=length, min_periods=1).sum()
    
    # Calculate CMF, handling division by zero
    cmf = pd.Series(0.0, index=high.index)
    valid_vol = vol_sum > 0
    cmf[valid_vol] = mfv_sum[valid_vol] / vol_sum[valid_vol]
    
    return cmf

def calculate_klinger_volume_oscillator(high, low, close, volume, fast=34, slow=55, signal=13):
    """Calculate Klinger Volume Oscillator (KVO) independently"""
    # Calculate typical price
    tp = (high + low + close) / 3
    
    # Calculate trend direction (1 for up, -1 for down)
    trend = pd.Series(0, index=tp.index)
    for i in range(1, len(tp)):
        trend.iloc[i] = 1 if tp.iloc[i] > tp.iloc[i-1] else -1
    
    # Calculate volume force
    vf = pd.Series(0.0, index=tp.index)
    
    for i in range(1, len(tp)):
        # High and low range
        hl_range = high.iloc[i] - low.iloc[i]
        
        # Only calculate if range is positive
        if hl_range > 0:
            # CM = ((C - L) - (H - C)) / (H - L)
            cm = ((close.iloc[i] - low.iloc[i]) - (high.iloc[i] - close.iloc[i])) / hl_range
            vf.iloc[i] = volume.iloc[i] * trend.iloc[i] * abs(cm) * 100
    
    # Calculate EMAs
    ema_fast = vf.ewm(span=fast, adjust=False).mean()
    ema_slow = vf.ewm(span=slow, adjust=False).mean()
    
    # Calculate KVO
    kvo = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = kvo.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = kvo - signal_line
    
    return {
        'kvo': kvo,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_volume_oscillator(volume, fast=14, slow=28):
    """Calculate Volume Oscillator independently"""
    # Calculate EMAs of volume
    ema_fast = volume.ewm(span=fast, adjust=False).mean()
    ema_slow = volume.ewm(span=slow, adjust=False).mean()
    
    # Calculate Volume Oscillator
    vo = ema_fast - ema_slow
    
    return vo

def calculate_volume_zone_oscillator(close, volume, length=14):
    """Calculate Volume Zone Oscillator (VZO) independently"""
    # Calculate volume EMAs
    ema_vol = volume.ewm(span=length, adjust=False).mean()
    
    # Separate volume by price direction
    vol_up = volume.copy()
    vol_down = volume.copy()
    
    for i in range(1, len(close)):
        if close.iloc[i] <= close.iloc[i-1]:
            vol_up.iloc[i] = 0
        else:
            vol_down.iloc[i] = 0
    
    # Calculate EMAs of up/down volume
    ema_vol_up = vol_up.ewm(span=length, adjust=False).mean()
    ema_vol_down = vol_down.ewm(span=length, adjust=False).mean()
    
    # Calculate VZO: ((Up Vol EMA - Down Vol EMA) / Vol EMA) * 100
    vzo = pd.Series(0.0, index=close.index)
    valid_denom = ema_vol > 0
    vzo[valid_denom] = 100 * (ema_vol_up[valid_denom] - ema_vol_down[valid_denom]) / ema_vol[valid_denom]
    
    return vzo

def calculate_volume_price_trend(close, volume):
    """Calculate Volume Price Trend (VPT) independently"""
    # Calculate price percent change
    pct_change = close.pct_change().fillna(0)
    
    # Calculate VPT
    vpt = pd.Series(0.0, index=close.index)
    
    # Initial value doesn't use pct_change
    vpt.iloc[0] = volume.iloc[0]
    
    # Calculate the rest of VPT values
    for i in range(1, len(close)):
        vpt.iloc[i] = vpt.iloc[i-1] + volume.iloc[i] * pct_change.iloc[i]
    
    return vpt

def calculate_volume_price_confirmation(close, volume):
    """Calculate Volume Price Confirmation indicator"""
    # Calculate price and volume direction
    price_dir = pd.Series(0, index=close.index)
    vol_dir = pd.Series(0, index=volume.index)
    
    for i in range(1, len(close)):
        # Price direction
        if close.iloc[i] > close.iloc[i-1]:
            price_dir.iloc[i] = 1
        elif close.iloc[i] < close.iloc[i-1]:
            price_dir.iloc[i] = -1
            
        # Volume direction
        if volume.iloc[i] > volume.iloc[i-1]:
            vol_dir.iloc[i] = 1
        elif volume.iloc[i] < volume.iloc[i-1]:
            vol_dir.iloc[i] = -1
    
    # Calculate confirmation (1 if direction matches, 0 otherwise)
    confirmation = pd.Series(0, index=close.index)
    
    for i in range(len(close)):
        if price_dir.iloc[i] != 0 and vol_dir.iloc[i] != 0:
            confirmation.iloc[i] = 1 if price_dir.iloc[i] == vol_dir.iloc[i] else 0
    
    return confirmation

def validate_volume_indicators(df, pair):
    """
    Validate volume indicators for a cryptocurrency pair
    
    Args:
        df: DataFrame with candle data
        pair: Symbol pair for context
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    issue_summary = {}
    
    # Skip if DataFrame is empty
    if df.empty:
        return {
            'pair': pair,
            'status': 'no_data',
            'issues_count': 0
        }
    
    # Check if required base columns exist
    required_base_columns = ['open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h']
    missing_base_columns = [col for col in required_base_columns if col not in df.columns]
    
    if missing_base_columns:
        return {
            'pair': pair,
            'status': 'missing_base_columns',
            'issues_count': len(missing_base_columns),
            'missing_columns': missing_base_columns
        }
    
    # Threshold for considering values as different (allowing for minor floating-point differences)
    # Increased thresholds for complex calculations
    obv_threshold = 1.0  # OBV can have larger differences due to integer rounding
    vwma_threshold = 0.5
    mfi_threshold = 0.5
    cmf_threshold = 0.01
    kvo_threshold = 1.0  # Klinger uses complex calculations with potential for larger differences
    vo_threshold = 1.0
    vzo_threshold = 1.0
    vpt_threshold = 1.0
    vpc_threshold = 0.01  # Should be very close, it's binary
    vol_change_threshold = 0.01
    
    # Validate OBV Slope
    if 'obv_slope_1h' in df.columns:
        # Calculate expected OBV and slope
        obv_results = calculate_obv(df['close_1h'], df['volume_1h'])
        expected_obv_slope = obv_results['obv_slope']
        
        # Calculate absolute differences
        obv_slope_diff = np.abs(df['obv_slope_1h'] - expected_obv_slope)
        obv_slope_issues = df[obv_slope_diff > obv_threshold]
        
        issue_count = len(obv_slope_issues)
        if issue_count > 0:
            issue_summary['obv_slope_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in obv_slope_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'obv_slope_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_obv_slope.loc[idx]) if not pd.isna(expected_obv_slope.loc[idx]) else None,
                    'actual': float(row['obv_slope_1h']),
                    'diff': float(obv_slope_diff.loc[idx]),
                    'details': f"OBV Slope calculation discrepancy"
                })
    
    # Validate Money Flow Index
    if 'money_flow_index_1h' in df.columns:
        # Calculate expected MFI
        expected_mfi = calculate_money_flow_index(
            df['high_1h'], df['low_1h'], df['close_1h'], df['volume_1h']
        )
        
        # Calculate absolute differences
        mfi_diff = np.abs(df['money_flow_index_1h'] - expected_mfi)
        mfi_issues = df[mfi_diff > mfi_threshold]
        
        issue_count = len(mfi_issues)
        if issue_count > 0:
            issue_summary['mfi_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in mfi_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'mfi_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_mfi.loc[idx]) if not pd.isna(expected_mfi.loc[idx]) else None,
                    'actual': float(row['money_flow_index_1h']),
                    'diff': float(mfi_diff.loc[idx]),
                    'details': f"Money Flow Index calculation discrepancy"
                })
    
    # Validate VWMA
    if 'vwma_20' in df.columns:
        # Calculate expected VWMA
        expected_vwma = calculate_vwma(df['close_1h'], df['volume_1h'])
        
        # Calculate absolute differences
        vwma_diff = np.abs(df['vwma_20'] - expected_vwma)
        vwma_issues = df[vwma_diff > vwma_threshold]
        
        issue_count = len(vwma_issues)
        if issue_count > 0:
            issue_summary['vwma_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in vwma_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'vwma_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_vwma.loc[idx]) if not pd.isna(expected_vwma.loc[idx]) else None,
                    'actual': float(row['vwma_20']),
                    'diff': float(vwma_diff.loc[idx]),
                    'details': f"VWMA calculation discrepancy"
                })
    
    # Validate Chaikin Money Flow
    if 'chaikin_money_flow' in df.columns:
        # Calculate expected CMF
        expected_cmf = calculate_chaikin_money_flow(
            df['high_1h'], df['low_1h'], df['close_1h'], df['volume_1h']
        )
        
        # Calculate absolute differences
        cmf_diff = np.abs(df['chaikin_money_flow'] - expected_cmf)
        cmf_issues = df[cmf_diff > cmf_threshold]
        
        issue_count = len(cmf_issues)
        if issue_count > 0:
            issue_summary['cmf_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in cmf_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'cmf_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_cmf.loc[idx]) if not pd.isna(expected_cmf.loc[idx]) else None,
                    'actual': float(row['chaikin_money_flow']),
                    'diff': float(cmf_diff.loc[idx]),
                    'details': f"Chaikin Money Flow calculation discrepancy"
                })
    
    # Validate Klinger Oscillator
    if 'klinger_oscillator' in df.columns:
        # Calculate expected KVO
        kvo_results = calculate_klinger_volume_oscillator(
            df['high_1h'], df['low_1h'], df['close_1h'], df['volume_1h']
        )
        expected_kvo = kvo_results['kvo']
        
        # Calculate absolute differences
        kvo_diff = np.abs(df['klinger_oscillator'] - expected_kvo)
        kvo_issues = df[kvo_diff > kvo_threshold]
        
        issue_count = len(kvo_issues)
        if issue_count > 0:
            issue_summary['kvo_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in kvo_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'kvo_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_kvo.loc[idx]) if not pd.isna(expected_kvo.loc[idx]) else None,
                    'actual': float(row['klinger_oscillator']),
                    'diff': float(kvo_diff.loc[idx]),
                    'details': f"Klinger Volume Oscillator calculation discrepancy"
                })
    
    # Validate Volume Oscillator
    if 'volume_oscillator' in df.columns:
        # Calculate expected Volume Oscillator
        expected_vo = calculate_volume_oscillator(df['volume_1h'])
        
        # Calculate absolute differences
        vo_diff = np.abs(df['volume_oscillator'] - expected_vo)
        vo_issues = df[vo_diff > vo_threshold]
        
        issue_count = len(vo_issues)
        if issue_count > 0:
            issue_summary['vo_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in vo_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'vo_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_vo.loc[idx]) if not pd.isna(expected_vo.loc[idx]) else None,
                    'actual': float(row['volume_oscillator']),
                    'diff': float(vo_diff.loc[idx]),
                    'details': f"Volume Oscillator calculation discrepancy"
                })
    
    # Validate Volume Zone Oscillator
    if 'volume_zone_oscillator' in df.columns:
        # Calculate expected VZO
        expected_vzo = calculate_volume_zone_oscillator(df['close_1h'], df['volume_1h'])
        
        # Calculate absolute differences
        vzo_diff = np.abs(df['volume_zone_oscillator'] - expected_vzo)
        vzo_issues = df[vzo_diff > vzo_threshold]
        
        issue_count = len(vzo_issues)
        if issue_count > 0:
            issue_summary['vzo_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in vzo_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'vzo_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_vzo.loc[idx]) if not pd.isna(expected_vzo.loc[idx]) else None,
                    'actual': float(row['volume_zone_oscillator']),
                    'diff': float(vzo_diff.loc[idx]),
                    'details': f"Volume Zone Oscillator calculation discrepancy"
                })
    
    # Validate Volume Price Trend
    if 'volume_price_trend' in df.columns:
        # Calculate expected VPT
        expected_vpt = calculate_volume_price_trend(df['close_1h'], df['volume_1h'])
        
        # Calculate absolute differences
        vpt_diff = np.abs(df['volume_price_trend'] - expected_vpt)
        vpt_issues = df[vpt_diff > vpt_threshold]
        
        issue_count = len(vpt_issues)
        if issue_count > 0:
            issue_summary['vpt_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in vpt_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'vpt_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_vpt.loc[idx]) if not pd.isna(expected_vpt.loc[idx]) else None,
                    'actual': float(row['volume_price_trend']),
                    'diff': float(vpt_diff.loc[idx]),
                    'details': f"Volume Price Trend calculation discrepancy"
                })
    
    # Validate Volume Price Confirmation
    if 'volume_price_confirmation' in df.columns:
        # Calculate expected Volume Price Confirmation
        expected_vpc = calculate_volume_price_confirmation(df['close_1h'], df['volume_1h'])
        
        # Calculate absolute differences
        vpc_diff = np.abs(df['volume_price_confirmation'] - expected_vpc)
        vpc_issues = df[vpc_diff > vpc_threshold]
        
        issue_count = len(vpc_issues)
        if issue_count > 0:
            issue_summary['vpc_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in vpc_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'vpc_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_vpc.loc[idx]) if not pd.isna(expected_vpc.loc[idx]) else None,
                    'actual': float(row['volume_price_confirmation']),
                    'diff': float(vpc_diff.loc[idx]),
                    'details': f"Volume Price Confirmation calculation discrepancy"
                })
    
    # Validate Volume Change Percentage
    if 'volume_change_pct_1h' in df.columns:
        # Calculate expected Volume Change Percentage
        expected_vol_change = df['volume_1h'].pct_change().fillna(0)
        
        # Calculate absolute differences
        vol_change_diff = np.abs(df['volume_change_pct_1h'] - expected_vol_change)
        vol_change_issues = df[vol_change_diff > vol_change_threshold]
        
        issue_count = len(vol_change_issues)
        if issue_count > 0:
            issue_summary['vol_change_issues'] = {'count': issue_count}
            
            # Record first few issues for reporting
            for idx, row in vol_change_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'vol_change_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': float(expected_vol_change.loc[idx]),
                    'actual': float(row['volume_change_pct_1h']),
                    'diff': float(vol_change_diff.loc[idx]),
                    'details': f"Volume Change Percentage calculation discrepancy"
                })
    
    # Check for directional consistency (only for significant price movements)
    directional_mismatch = {'count': 0}
    
    # Check if OBV Slope matches price direction for significant price movements
    if 'obv_slope_1h' in df.columns:
        # Calculate price change
        price_change = df['close_1h'].pct_change()
        
        # Find significant price movements (more than 1%)
        significant_moves = df[abs(price_change) > 0.01]
        
        # Track significant mismatches
        mismatch_count = 0
        
        # Check if OBV slope direction matches price direction
        for idx, row in significant_moves.iterrows():
            price_dir = np.sign(price_change.loc[idx])
            obv_slope_dir = np.sign(row['obv_slope_1h'])
            
            # Mismatch in direction (should generally move in same direction)
            if price_dir * obv_slope_dir < 0:
                mismatch_count += 1
                
                # Only record first few mismatches for reporting
                if mismatch_count <= 5:
                    issues.append({
                        'issue_type': 'directional_mismatch',
                        'timestamp': row['timestamp_utc'],
                        'price_change': float(price_change.loc[idx]),
                        'obv_slope': float(row['obv_slope_1h']),
                        'details': f"OBV Slope direction doesn't match significant price movement direction"
                    })
        
        if mismatch_count > 0:
            directional_mismatch['count'] = mismatch_count
    
    # Add directional mismatch to the issue summary if there are any
    if directional_mismatch['count'] > 0:
        issue_summary['directional_mismatch'] = directional_mismatch
    
    # Calculate total issues
    total_issues = sum(category['count'] for category in issue_summary.values())
    
    # Calculate issue percentage based on number of candles
    issue_percentage = (total_issues / len(df)) * 100 if len(df) > 0 else 0
    
    return {
        'pair': pair,
        'status': 'completed',
        'issues_count': total_issues,
        'candles_count': len(df),
        'issue_percentage': issue_percentage,
        'issue_summary': issue_summary,
        'issues': issues
    }

if __name__ == "__main__":
    main_validator(validate_volume_indicators, "Volume Indicators Validator", 
                  "Validates volume technical indicators by recomputing and comparing them to stored values")