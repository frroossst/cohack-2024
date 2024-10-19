import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft

n_terms = 1_000_000  # Number of FFT terms to keep (you can adjust this for smoother/finer fit)
n_terms = 2_500


# Read data from gpsdata.csv
latitudes = []
longitudes = []

with open('/home/home/Desktop/Projects/pawpatrol/data/gpsdata.csv', 'r') as file:
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
# plt.show()


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
    print(f"Term {i}: {coef.real:.3f} cos({i} * x) + {coef.imag:.3f} sin({i} * x)")
    continue

# First, let's examine the coefficients
print("First few coefficients magnitudes:")
for i in range(min(10, len(coefficients))):
    print(f"Component {i}: {np.abs(coefficients[i]):.2e}")

def plot_fourier_components_improved(coefficients, num_components_to_plot=5):
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Generate x values for plotting
    x = np.linspace(0, 2*np.pi, 1000)
    
    # Skip the DC component (index 0) when plotting individual waves
    for i in range(1, min(num_components_to_plot + 1, len(coefficients))):
        # Extract amplitude and phase from complex coefficient
        amplitude = np.abs(coefficients[i])
        phase = np.angle(coefficients[i])
        
        # Generate the component wave without normalizing to see actual contribution
        wave = np.real(amplitude * np.exp(1j * (i * x + phase)))
        
        # Plot on first subplot
        ax1.plot(x, wave, label=f'Component {i}')
    
    ax1.set_title('Individual Fourier Components (Actual Scale)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.legend()
    
    # Plot combined wave (excluding DC component)
    combined_wave = np.zeros_like(x)
    for i in range(1, len(coefficients)):
        amplitude = np.abs(coefficients[i])
        phase = np.angle(coefficients[i])
        combined_wave += np.real(amplitude * np.exp(1j * (i * x + phase)))
    
    ax2.plot(x, combined_wave, 'r-', label='Combined Wave (AC Components)')
    ax2.set_title('Combined Fourier Series (Without DC Component)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.legend()
    
    # Plot frequency spectrum (magnitude of coefficients)
    frequencies = np.arange(len(coefficients))
    magnitudes = np.abs(coefficients)
    
    # Plot in log scale to better see the distribution
    ax3.semilogy(frequencies[1:], magnitudes[1:], 'b.')  # Skip DC component
    ax3.set_title('Frequency Spectrum (Log Scale)')
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Magnitude (log scale)')
    ax3.grid(True)
    
    plt.tight_layout()
    return fig

# Create the visualization
fig = plot_fourier_components_improved(coefficients, num_components_to_plot=10)
plt.show()

# Let's also look at the relative contributions of components
magnitudes = np.abs(coefficients)
total_power = np.sum(magnitudes**2)
relative_power = (magnitudes**2 / total_power) * 100

print("\nRelative power contribution of first 10 components:")
for i in range(min(10, len(coefficients))):
    print(f"Component {i}: {relative_power[i]:.2f}%")

# Plot relative power contribution
plt.figure(figsize=(12, 6))
plt.bar(range(min(20, len(coefficients))), relative_power[:20])
plt.title('Relative Power Contribution of First 20 Components')
plt.xlabel('Component Index')
plt.ylabel('Relative Power (%)')
plt.grid(True)
plt.show()


