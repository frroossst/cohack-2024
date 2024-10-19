import numpy as np

import numpy as np
import pandas as pd

def conc(df, lat_col='latitude', lon_col='longitude', threshold=0.0005):
    """
    Condenses datapoints in a DataFrame that are closer than the given threshold.

    Parameters:
        df (pd.DataFrame): A DataFrame containing latitude and longitude columns.
        lat_col (str): The name of the column containing latitude values.
        lon_col (str): The name of the column containing longitude values.
        threshold (float): The maximum distance between points to consider them as "close".

    Returns:
        pd.DataFrame: A DataFrame with condensed datapoints.
    """
    if df.empty:
        return pd.DataFrame(columns=[lat_col, lon_col])

    # Extract latitude and longitude
    points = df[[lat_col, lon_col]].values
    
    # List to store condensed points
    condensed_points = []
    
    # Keep track of processed indices
    processed = np.zeros(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        if processed[i]:
            continue

        # Find all points that are within the threshold
        close_points = np.linalg.norm(points - point, axis=1) < threshold
        
        # Calculate the mean of the close points
        condensed_point = np.mean(points[close_points], axis=0)
        
        # Add the condensed point to the result
        condensed_points.append(condensed_point)
        
        # Mark close points as processed
        processed[close_points] = True

    # Create a DataFrame from the condensed points
    return pd.DataFrame(condensed_points, columns=[lat_col, lon_col])

