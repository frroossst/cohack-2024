import pandas as pd
import numpy as np
from scipy.spatial import distance

def load_data(file_path):
    """Load the latitude and longitude data from a CSV file."""
    return pd.read_csv(file_path)

def calculate_outliers(df, lat_col='latitude', lon_col='longitude'):
    """Identify outliers in latitude and longitude data."""
    # Calculate the mean and standard deviation
    lat_mean = df[lat_col].mean()
    lon_mean = df[lon_col].mean()
    
    lat_std = df[lat_col].std()
    lon_std = df[lon_col].std()

    # Calculate z-scores
    df['lat_z'] = (df[lat_col] - lat_mean) / lat_std
    df['lon_z'] = (df[lon_col] - lon_mean) / lon_std

    # Identify outliers based on z-score
    df['outlier'] = (df['lat_z'].abs() > 3) | (df['lon_z'].abs() > 3)

    return df

def main():
    file_path = '/home/home/Desktop/Projects/pawpatrol/data/cutesiedata.csv'  # Replace with your CSV file path
    df = load_data(file_path)
    
    # Assuming the CSV has columns named 'latitude' and 'longitude'
    outlier_df = calculate_outliers(df)

    # Print or save outliers
    outliers = outlier_df[outlier_df['outlier']]
    if not outliers.empty:
        print("Outliers detected:")
        print(outliers)
    else:
        print("No outliers detected.")

if __name__ == "__main__":
    main()

