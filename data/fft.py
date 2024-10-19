import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft

n_terms = 1_000_000  # Number of FFT terms to keep (you can adjust this for smoother/finer fit)
n_terms = 2_500


# Read data from gpsdata.csv
latitudes = []
longitudes = []

with open('gpsdata.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        latitudes.append(float(row[1]))
        longitudes.append(float(row[2]))

# Convert to numpy arrays
latitudes = np.array(latitudes)
longitudes = np.array(longitudes)

# Perform FFT on the data (we assume latitudes as the dependent variable on longitudes)
fft_latitudes = fft(latitudes)

# Truncate higher-order terms to get a smoother line of best fit (keep only a few terms)
fft_latitudes_truncated = np.zeros_like(fft_latitudes)
fft_latitudes_truncated[:n_terms] = fft_latitudes[:n_terms]

# Perform inverse FFT to get the smoothed latitudes
smoothed_latitudes = ifft(fft_latitudes_truncated).real

# Plot the line of best fit
plt.plot(longitudes, smoothed_latitudes, color='green', label='FFT Best Fit')
# original data
plt.scatter(longitudes, latitudes, color='blue', label='Original Data', alpha=0.5)

# Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Line of Best Fit using FFT')

# Show the plot
plt.legend()
plt.show()


# Function to check if FFT compression is effective
def check_if_compressed(original_data, compressed_data, n_terms, total_data_points):
    # 1. Mean Squared Error (MSE)
    mse = np.mean((original_data - compressed_data) ** 2)

    # 2. Compression Ratio
    original_size = total_data_points  # Original data points size
    compressed_size = n_terms * 2  # Each complex number has real and imaginary parts
    compression_ratio = original_size / compressed_size

    # 3. Data Savings
    original_storage_bytes = original_size * 8  # Assuming each float is 8 bytes
    compressed_storage_bytes = compressed_size * 8  # Storing real and imaginary parts
    data_saved = (1 - (compressed_storage_bytes / original_storage_bytes)) * 100

    # Display statistics
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Data Saved: {data_saved:.2f}%")
    
    return mse, compression_ratio, data_saved

check_if_compressed(latitudes, smoothed_latitudes, n_terms, len(latitudes))

# Print the best fit as a series of Fourier components
coefficients = fft_latitudes[:n_terms]
print("Equation of the best fit (Fourier Series):")
for i, coef in enumerate(coefficients):
    # print(f"Term {i}: {coef.real:.3f} cos({i} * x) + {coef.imag:.3f} sin({i} * x)")
    continue


