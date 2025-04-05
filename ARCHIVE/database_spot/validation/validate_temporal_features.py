#!/usr/bin/env python3
"""
Temporal Features Validator
- Validates: hour_of_day, day_of_week, month_of_year, is_weekend, asian_session, etc.
- Recalculates from timestamp_utc
- Ensures time zone handling is consistent with feature processor
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from database_spot.validation.validation_utils import main_validator

def validate_temporal_features(df, pair):
    """
    Validate temporal features for a cryptocurrency pair
    
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
    
    # Check if required timestamp column exists
    if 'timestamp_utc' not in df.columns:
        return {
            'pair': pair,
            'status': 'missing_timestamp',
            'issues_count': 1,
            'issues': [{'issue_type': 'missing_column', 'column': 'timestamp_utc'}]
        }
    
    # List of temporal features to validate
    temporal_features = [
        'hour_of_day', 'day_of_week', 'month_of_year', 
        'is_weekend', 'asian_session', 'european_session', 'american_session'
    ]
    
    # Check which features are present in the dataframe
    present_features = [col for col in temporal_features if col in df.columns]
    
    if not present_features:
        return {
            'pair': pair,
            'status': 'no_temporal_features',
            'issues_count': 0
        }
    
    # Initialize issue counters for each feature
    for feature in present_features:
        issue_summary[f'{feature}_issues'] = {'count': 0}
    
    # Convert timestamps to UTC before extracting components
    # The feature calculation logic likely used UTC timestamps
    timestamps = pd.to_datetime(df['timestamp_utc']).dt.tz_convert('UTC')
    
    # Hour of day (0-23) in UTC
    if 'hour_of_day' in present_features:
        expected_hour = timestamps.dt.hour
        hour_issues = df[df['hour_of_day'] != expected_hour]
        
        if not hour_issues.empty:
            issue_summary['hour_of_day_issues']['count'] = len(hour_issues)
            
            # Record first few issues for reporting
            for idx, row in hour_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'hour_of_day_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': int(expected_hour.loc[idx]),
                    'actual': int(row['hour_of_day']),
                    'details': f"Hour of day mismatch"
                })
    
    # Day of week (0-6, Monday=0)
    if 'day_of_week' in present_features:
        expected_day = timestamps.dt.dayofweek
        day_issues = df[df['day_of_week'] != expected_day]
        
        if not day_issues.empty:
            issue_summary['day_of_week_issues']['count'] = len(day_issues)
            
            # Record first few issues for reporting
            for idx, row in day_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'day_of_week_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': int(expected_day.loc[idx]),
                    'actual': int(row['day_of_week']),
                    'details': f"Day of week mismatch"
                })
    
    # Month of year (1-12)
    if 'month_of_year' in present_features:
        expected_month = timestamps.dt.month
        month_issues = df[df['month_of_year'] != expected_month]
        
        if not month_issues.empty:
            issue_summary['month_of_year_issues']['count'] = len(month_issues)
            
            # Record first few issues for reporting
            for idx, row in month_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'month_of_year_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': int(expected_month.loc[idx]),
                    'actual': int(row['month_of_year']),
                    'details': f"Month of year mismatch"
                })
    
    # Is weekend (Friday after 9 PM UTC or Saturday or Sunday)
    if 'is_weekend' in present_features:
        day_of_week = timestamps.dt.dayofweek
        hour_of_day = timestamps.dt.hour
        
        # Weekend is defined as Friday after 21:00 UTC, or all day Saturday/Sunday
        expected_weekend = ((day_of_week == 4) & (hour_of_day >= 21)) | (day_of_week >= 5)
        expected_weekend = expected_weekend.astype(int)
        
        weekend_issues = df[df['is_weekend'] != expected_weekend]
        
        if not weekend_issues.empty:
            issue_summary['is_weekend_issues']['count'] = len(weekend_issues)
            
            # Record first few issues for reporting
            for idx, row in weekend_issues.head(5).iterrows():
                issues.append({
                    'issue_type': 'is_weekend_issue',
                    'timestamp': row['timestamp_utc'],
                    'expected': int(expected_weekend.loc[idx]),
                    'actual': int(row['is_weekend']),
                    'details': f"Weekend flag mismatch"
                })
    
    # Trading sessions
    # Define session times in UTC (matches feature processor)
    session_hours = {
        'asian_session': {'start': 0, 'end': 8},
        'european_session': {'start': 8, 'end': 16},
        'american_session': {'start': 14, 'end': 22}
    }
    
    for session, hours in session_hours.items():
        if session in present_features:
            start_hour = hours['start']
            end_hour = hours['end']
            
            # Calculate if the hour falls within session hours
            expected_session = ((hour_of_day >= start_hour) & (hour_of_day < end_hour)).astype(int)
            session_issues = df[df[session] != expected_session]
            
            if not session_issues.empty:
                issue_summary[f'{session}_issues']['count'] = len(session_issues)
                
                # Record first few issues for reporting
                for idx, row in session_issues.head(5).iterrows():
                    issues.append({
                        'issue_type': f'{session}_issue',
                        'timestamp': row['timestamp_utc'],
                        'expected': int(expected_session.loc[idx]),
                        'actual': int(row[session]),
                        'hour': int(hour_of_day.loc[idx]),
                        'details': f"{session.replace('_', ' ').title()} flag mismatch"
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
    main_validator(validate_temporal_features, "Temporal Features Validator", 
                  "Validates time-based features by recalculating them from timestamp_utc")