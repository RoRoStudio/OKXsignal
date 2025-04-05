#!/usr/bin/env python3
"""
Volume Indicators Validator
- Validates: obv_slope, volume_change_pct, volume_zone_oscillator, chaikin_money_flow, klinger, vwma, etc.
- Confirms directional change logic matches price/volume shifts
"""

import pandas as pd
import numpy as np
from database_spot.validation.validation_utils import main_validator

def calculate_obv(close, volume):
    """Calculate OBV to match feature_processor implementation"""
    n = len(close)
    obv = np.zeros(n)
    
    # First OBV is first volume
    if n > 0:
        obv[0] = volume.iloc[0]
        
        # Calculate rest of OBV
        for i in range(1, n):
            if close.iloc[i] > close.iloc[i-1]:
                obv[i] = obv[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv[i] = obv[i-1] - volume.iloc[i]
            else:
                obv[i] = obv[i-1]
    
    # Convert to Series for compatibility with validation code
    obv_series = pd.Series(obv, index=close.index)
    
    # Calculate OBV slope
    obv_slope = obv_series.diff(3) / 3
    
    return {
        'obv': obv_series,
        'obv_slope': obv_slope
    }

def calculate_money_flow_index(high, low, close, volume, length=14):
    """Calculate MFI to match feature_processor implementation"""
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate money flow
    money_flow = typical_price * volume
    
    # Calculate positive and negative money flow
    pos_flow = np.zeros(len(typical_price))
    neg_flow = np.zeros(len(typical_price))
    
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            pos_flow[i] = money_flow.iloc[i]
        else:
            neg_flow[i] = money_flow.iloc[i]
    
    # Convert to Series
    pos_flow = pd.Series(pos_flow, index=typical_price.index)
    neg_flow = pd.Series(neg_flow, index=typical_price.index)
    
    # Calculate positive and negative flow sums over the period
    pos_sum = pos_flow.rolling(window=length, min_periods=1).sum()
    neg_sum = neg_flow.rolling(window=length, min_periods=1).sum()
    
    # Calculate money ratio and handle division by zero
    money_ratio = np.zeros(len(pos_sum))
    for i in range(len(pos_sum)):
        if neg_sum.iloc[i] != 0:
            money_ratio[i] = pos_sum.iloc[i] / neg_sum.iloc[i]
        else:
            money_ratio[i] = 100  # Handle division by zero
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + money_ratio))
    
    return pd.Series(mfi, index=high.index).fillna(50)

def calculate_vwma(close, volume, length=20):
    """Calculate VWMA to match feature_processor implementation"""
    weighted_sum = (close * volume).rolling(window=length, min_periods=1).sum()
    vol_sum = volume.rolling(window=length, min_periods=1).sum()
    
    # Avoid division by zero
    vwma = close.copy()
    mask = vol_sum > 0
    vwma[mask] = weighted_sum[mask] / vol_sum[mask]
    
    return vwma

def calculate_chaikin_money_flow(high, low, close, volume, length=20):
    """Calculate CMF to match feature_processor implementation"""
    # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    high_low_range = high - low
    
    # Avoid division by zero
    money_flow_multiplier = pd.Series(0.0, index=high.index)
    mask = high_low_range > 0
    money_flow_multiplier[mask] = ((close[mask] - low[mask]) - (high[mask] - close[mask])) / high_low_range[mask]
    
    # Money Flow Volume = Money Flow Multiplier * Volume
    money_flow_volume = money_flow_multiplier * volume
    
    # Chaikin Money Flow = Sum(Money Flow Volume) / Sum(Volume) over period
    cmf = pd.Series(0.0, index=high.index)
    
    # Calculate sums
    mfv_sum = money_flow_volume.rolling(window=length, min_periods=1).sum()
    vol_sum = volume.rolling(window=length, min_periods=1).sum()
    
    # Avoid division by zero
    mask = vol_sum > 0
    cmf[mask] = mfv_sum[mask] / vol_sum[mask]
    
    return cmf

def calculate_klinger_oscillator(high, low, close, volume, fast=34, slow=55, signal=13):
    """Calculate KVO to match feature_processor implementation"""
    # Calculate trend direction (1 for up, -1 for down)
    tp = (high + low + close) / 3
    trend = pd.Series(0, index=tp.index)
    
    for i in range(1, len(tp)):
        trend.iloc[i] = 1 if tp.iloc[i] > tp.iloc[i-1] else -1
    
    # Calculate volume force
    vf = pd.Series(0.0, index=volume.index)
    
    for i in range(1, len(volume)):
        # High-low range
        hl_range = high.iloc[i] - low.iloc[i]
        
        # Avoid division by zero
        if hl_range > 0:
            cm = ((close.iloc[i] - low.iloc[i]) - (high.iloc[i] - close.iloc[i])) / hl_range
            vf.iloc[i] = volume.iloc[i] * trend.iloc[i] * abs(cm) * 2
    
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
    """Calculate Volume Oscillator to match feature_processor implementation"""
    # Calculate EMAs
    ema_fast = volume.ewm(span=fast, adjust=False).mean()
    ema_slow = volume.ewm(span=slow, adjust=False).mean()
    
    # Return oscillator
    return ema_fast - ema_slow

def calculate_volume_zone_oscillator(close, volume, length=14):
    """Calculate VZO to match feature_processor implementation"""
    # Calculate volume EMAs
    ema_vol = volume.ewm(span=length, adjust=False).mean()
    
    # Separate volumes by price direction
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
    
    # Calculate VZO
    vzo = pd.Series(0.0, index=close.index)
    mask = ema_vol > 0
    vzo[mask] = 100 * (ema_vol_up[mask] - ema_vol_down[mask]) / ema_vol[mask]
    
    return vzo

def calculate_volume_price_trend(close, volume):
    """Calculate VPT to match feature_processor implementation"""
    vpt = pd.Series(0.0, index=close.index)
    
    # First value is first volume
    vpt.iloc[0] = volume.iloc[0]
    
    # Calculate price change and VPT
    for i in range(1, len(close)):
        if close.iloc[i-1] > 0:  # Avoid division by zero
            pct_change = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            vpt.iloc[i] = vpt.iloc[i-1] + volume.iloc[i] * pct_change
    
    return vpt

def calculate_volume_price_confirmation(close, volume):
    """Calculate Volume Price Confirmation to match feature_processor"""
    vpc = pd.Series(0, index=close.index)
    
    # First value is always 0
    vpc.iloc[0] = 0
    
    # Calculate direction match
    for i in range(1, len(close)):
        close_dir = np.sign(close.iloc[i] - close.iloc[i-1])
        vol_dir = np.sign(volume.iloc[i] - volume.iloc[i-1])
        
        # 1 if directions match, 0 otherwise
        vpc.iloc[i] = 1 if close_dir == vol_dir and close_dir != 0 and vol_dir != 0 else 0
    
    return vpc

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
    
    # Threshold for considering values as different
    threshold = 1e-6
    
    # Validate OBV Slope
    if 'obv_slope_1h' in df.columns:
        # Calculate expected OBV and slope
        obv_results = calculate_obv(df['close_1h'], df['volume_1h'])
        expected_obv_slope = obv_results['obv_slope']
        
        # Calculate absolute differences
        obv_slope_diff = np.abs(df['obv_slope_1h'] - expected_obv_slope)
        obv_slope_issues = df[obv_slope_diff > threshold]
        
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
        mfi_issues = df[mfi_diff > threshold]
        
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
        vwma_issues = df[vwma_diff > threshold]
        
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
        cmf_issues = df[cmf_diff > threshold]
        
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
        kvo_results = calculate_klinger_oscillator(
            df['high_1h'], df['low_1h'], df['close_1h'], df['volume_1h']
        )
        expected_kvo = kvo_results['kvo']
        
        # Calculate absolute differences
        kvo_diff = np.abs(df['klinger_oscillator'] - expected_kvo)
        kvo_issues = df[kvo_diff > threshold]
        
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
        vo_issues = df[vo_diff > threshold]
        
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
        vzo_issues = df[vzo_diff > threshold]
        
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
        vpt_issues = df[vpt_diff > threshold]
        
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
        vpc_issues = df[vpc_diff > threshold]
        
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
        vol_change_issues = df[vol_change_diff > threshold]
        
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