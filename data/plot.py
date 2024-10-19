import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class DataSamples:
    def __init__(self):
        self.data = []  # Initialize as an empty list

    def len(self):
        return len(self.data)

    def set(self, data):
        self.data = data  # Expect data to be a list of tuples (latitude, longitude)

    def get(self):
        return self.data


# Haversine formula to calculate distance between two GPS points
def haversine(coord1, coord2):
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # Distance in meters
    return distance


# Function to calculate the angle between two points in degrees
def calculate_angle(p1, p2):
    delta_y = p2[0] - p1[0]
    delta_x = p2[1] - p1[1]
    angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)  # Convert radians to degrees
    return angle


def read_data():
    data = pd.read_csv('/home/home/Desktop/Projects/pawpatrol/data/gpsdata.csv', header=None, names=['timestamp', 'latitude', 'longitude'])
    # Convert ataFrame to list of tuples (latitude, longitude)
    return list(zip(data['latitude'], data['longitude']))


def original() -> DataSamples:
    samples = DataSamples()

    data = read_data()
    total_points = len(data)

    samples.set(data)  # Store data as a list of tuples
    return samples


###############################################################################
#                                                                            #
# Linear Path Optimization                                                   #
#                                                                            #
###############################################################################

def linear_path_optimization(data: DataSamples) -> DataSamples:
    samples = DataSamples()
    data = read_data()
    samples.set(data)  
    total_points = samples.len()  # Get length from the DataSamples class

    filtered_latitudes = []
    filtered_longitudes = []

    # First pass: Horizontal optimization
    threshold_angle = 5  # You can adjust this threshold

    # Collect the first point
    filtered_latitudes.append(samples.get()[0][0])
    filtered_longitudes.append(samples.get()[0][1])

    for i in range(1, total_points):
        prev_point = samples.get()[i - 1]
        curr_point = samples.get()[i]

        angle = calculate_angle(prev_point, curr_point)

        if abs(angle) > threshold_angle:
            filtered_latitudes.append(curr_point[0])
            filtered_longitudes.append(curr_point[1])

    # Second pass: Vertical optimization
    threshold_distance = 1.0  # You can adjust this threshold

    # Collect the first point
    filtered_latitudes.append(samples.get()[0][0])

    for i in range(1, len(filtered_latitudes)):
        prev_point = (filtered_latitudes[i - 1], filtered_longitudes[i - 1])
        curr_point = (filtered_latitudes[i], filtered_longitudes[i])

        distance = haversine(prev_point, curr_point)

        if distance > threshold_distance:
            filtered_longitudes.append(curr_point[1])

    # Combine latitudes and longitudes into a list of tuples
    filtered_data = list(zip(filtered_latitudes, filtered_longitudes))
    samples.set(filtered_data)  # Set the filtered data back to samples

    return samples


###############################################################################
#                                                                            #
# Threshold Bubble Optimization                                              #
#                                                                            #
###############################################################################
def threshold_bubble_optimization(data: DataSamples) -> DataSamples:
    samples = DataSamples()
    data = read_data()
    samples.set(data)  
    total_points = samples.len()  # Get length from the DataSamples class

    # Initialize lists to hold filtered points
    filtered_latitudes = []
    filtered_longitudes = []

    # Thresholds
    threshold_angle = 5  # Angle deviation threshold in degrees
    min_distance = 1.0  # Minimum distance in meters

    # Collect the first point
    filtered_latitudes.append(samples.get()[0][0])
    filtered_longitudes.append(samples.get()[0][1])

    # Iterate through the data points
    for i in range(1, total_points):
        prev_point = samples.get()[i - 1]
        curr_point = samples.get()[i]

        # Calculate the angle between the previous and current point
        angle = calculate_angle(prev_point, curr_point)

        # Calculate the distance between the previous and current point
        distance = haversine(prev_point, curr_point)

        # If the angle deviation exceeds the threshold or the distance exceeds the minimum distance, collect the current point
        if abs(angle) > threshold_angle or distance > min_distance:
            filtered_latitudes.append(curr_point[0])
            filtered_longitudes.append(curr_point[1])

    # Combine latitudes and longitudes into a list of tuples
    filtered_data = list(zip(filtered_latitudes, filtered_longitudes))
    samples.set(filtered_data)  # Set the filtered data back to samples

    return samples


###############################################################################
# OPT 1 and then OPT 2 #
###############################################################################
def opt1_opt2(data: DataSamples) -> DataSamples:
    samples = DataSamples()
    data = read_data()
    samples.set(data)  
    total_points = samples.len()  # Get length from the DataSamples class

    # Apply linear path optimization
    samples = DataSamples()
    data = read_data()
    samples.set(data)  

    opt1 = linear_path_optimization(samples)
    opt1_points = opt1.len()  # Get length from optimized data

    # Apply threshold bubble optimization
    samples = DataSamples()
    data = read_data()
    samples.set(data)  

    opt2 = threshold_bubble_optimization(samples)
    opt2_points = opt2.len()  # Get length from optimized data

    # Combine both optimizations
    samples = DataSamples()
    data = read_data()
    samples.set(data)
    
    only_opt1 = opt1.get()
    opt1_stacked_opt2 = DataSamples()
    opt1_stacked_opt2.set(only_opt1)
    opt1_stacked_opt2 = threshold_bubble_optimization(opt1_stacked_opt2)

    both = DataSamples()
    both.set(opt1_stacked_opt2.get())
    both_points = both.len()  # Get length from optimized data

    # print out statistics
    print(f'Original Total Points: {total_points}, percentage of points kept: {((total_points-total_points)/total_points) * 100}%')
    print(f'Linear Path Optimization Total Points: {opt1_points}, percent reduction : {((total_points-opt1_points)/total_points) * 100}%')
    print(f'Threshold Bubble Optimization Total Points: {opt2_points}, percent reduction : {((total_points-opt2_points)/total_points) * 100}%')
    print(f'Both Optimizations Total Points: {both_points}', f'percent reduction : {((total_points-both_points)/total_points) * 100}%')


    return both


###############################################################################
#                                                                            #
# Plotting                                                                   #
#                                                                            #
###############################################################################

def plot(data: DataSamples, title: str):
    latitudes, longitudes = zip(*data.get())  # Unzip into latitudes and longitudes
    plt.figure(figsize=(10, 6))
    plt.plot(longitudes, latitudes, color='red', marker='o', linestyle='-', markersize=2)
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid()
    plt.show()


def plot_all(data: DataSamples, opt1: DataSamples, opt2: DataSamples, both: DataSamples):
    original_latitudes, original_longitudes = zip(*data.get())
    opt1_latitudes, opt1_longitudes = zip(*opt1.get())
    opt2_latitudes, opt2_longitudes = zip(*opt2.get())
    both_latitudes, both_longitudes = zip(*both.get())

    plt.figure(figsize=(10, 6))
    plt.plot(original_longitudes, original_latitudes, color='grey', marker='o', linestyle='-', markersize=2, label='Original')
    plt.plot(opt1_longitudes, opt1_latitudes, color='blue', marker='o', linestyle='-', markersize=2, label='Linear Path Optimization')
    plt.plot(opt2_longitudes, opt2_latitudes, color='green', marker='o', linestyle='-', markersize=2, label='Threshold Bubble Optimization')
    plt.plot(both_longitudes, both_latitudes, color='red', marker='o', linestyle='-', markersize=2, label='Both Optimizations')
    plt.title('GPS Data Optimization')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    data = original()
    opt1 = linear_path_optimization(data)
    opt2 = threshold_bubble_optimization(data)
    both = opt1_opt2(data)

    plot(data, 'Original GPS Data')
    plot(opt1, 'Optimized GPS Data (Linear Path Optimization)')
    plot(opt2, 'Optimized GPS Data (Threshold Bubble Optimization)')
    plot(both, 'Optimized GPS Data (Both Optimizations)')

    plot_all(data, opt1, opt2, both)


if __name__ == "__main__":
    main()
