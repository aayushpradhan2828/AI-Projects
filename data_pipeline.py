import pandas as pd
import numpy as np

# AI Data Pipeline — Aayush Pradhan
# Demonstrates feature engineering and data preprocessing

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def preprocess(df):
    """Clean and deduplicate data."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def feature_engineering(df, target_col):
    """Apply normalization and feature extraction."""
    df['normalized'] = (df[target_col] - df[target_col].mean()) / df[target_col].std()
    df['rolling_mean'] = df[target_col].rolling(window=3).mean()
    return df

def anomaly_detection(df, target_col, threshold=2.0):
    """Flag anomalies beyond threshold standard deviations."""
    mean = df[target_col].mean()
    std = df[target_col].std()
    df['anomaly'] = np.abs(df[target_col] - mean) > threshold * std
    return df

def run_pipeline(filepath, target_col):
    """Execute full data pipeline."""
    df = load_data(filepath)
    df = preprocess(df)
    df = feature_engineering(df, target_col)
    df = anomaly_detection(df, target_col)
    print(f"Pipeline complete. Rows processed: {len(df)}")
    print(f"Anomalies detected: {df['anomaly'].sum()}")
    return df
