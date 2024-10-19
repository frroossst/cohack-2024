import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# Step 1: Load GPS data from CSV
gps_data = pd.read_csv('gpsdata.csv', header=None, names=['timestamp', 'latitude', 'longitude'])

# Convert timestamps to a numerical format (seconds)
timestamps = pd.to_datetime(gps_data['timestamp'])
time_numeric = (timestamps - timestamps.min()).dt.total_seconds()

# Step 2: Remove duplicate timestamps
gps_data_unique = gps_data.drop_duplicates(subset='timestamp')
timestamps_unique = pd.to_datetime(gps_data_unique['timestamp'])
time_numeric_unique = (timestamps_unique - timestamps_unique.min()).dt.total_seconds()

# Step 3: Apply Fourier Transform to latitude and longitude
lat_transform = np.fft.fft(gps_data_unique['latitude'])
long_transform = np.fft.fft(gps_data_unique['longitude'])

# Step 4: Reconstruct the signal using a subset of coefficients (for compression)
num_coefficients = 50  # Adjust this number for different compression levels
lat_reconstructed = np.fft.ifft(lat_transform[:num_coefficients])
long_reconstructed = np.fft.ifft(long_transform[:num_coefficients])

# Step 5: Interpolate for a smooth curve
interp_lat = interp1d(time_numeric_unique[:len(lat_reconstructed)], lat_reconstructed.real, kind='cubic')
interp_long = interp1d(time_numeric_unique[:len(long_reconstructed)], long_reconstructed.real, kind='cubic')

# Create a finer time grid for plotting
fine_time = np.linspace(time_numeric_unique.min(), time_numeric_unique.max(), num=1000)

# Get interpolated values for smooth curve
smooth_lat = interp_lat(fine_time)
smooth_long = interp_long(fine_time)

# Step 6: Plot the original and smooth reconstructed curves
plt.figure(figsize=(12, 6))
plt.plot(gps_data_unique['longitude'], gps_data_unique['latitude'], 'bo-', label='Original Data')
plt.plot(smooth_long, smooth_lat, 'ro-', label='Smooth Reconstructed Data')
plt.title('Original vs. Reconstructed GPS Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid()
plt.show()

# Step 7: Save the transformed data to a new CSV
transformed_data = pd.DataFrame({
    'lat_real': lat_transform.real,
    'lat_imag': lat_transform.imag,
    'long_real': long_transform.real,
    'long_imag': long_transform.imag
})
transformed_data.to_csv('transformed_gpsdata.csv', index=False)

# Step 8: Calculate the size of the original and transformed data in kilobytes
original_size = os.path.getsize('gpsdata.csv') / 1024  # Size in KB
transformed_size = os.path.getsize('transformed_gpsdata.csv') / 1024  # Size in KB

# Print out the sizes
print(f'Original CSV size: {original_size:.2f} KB')
print(f'Transformed CSV size: {transformed_size:.2f} KB')

