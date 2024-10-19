import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# Haversine formula to calculate distance between two GPS points
def haversine(coord1, coord2):
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # Distance in meters
    return distance

# Read the GPS data from the CSV file
data = pd.read_csv('/home/home/Desktop/Projects/pawpatrol/data/gpsdata.csv', header=None, names=['timestamp', 'latitude', 'longitude'])

# Count the total number of points
total_points = len(data)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(data['longitude'], data['latitude'], color='blue', marker='o')
plt.title('GPS Coordinates Plot')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.axis('equal')  # Equal aspect ratio ensures the plot is not distorted

# Show total points on the graph
plt.text(0.05, 0.95, f'Total Points: {total_points}', transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='lightgray'))

plt.show()

# Print the total number of points in the console
print(f'Total number of GPS points: {total_points}')


###############################################################################
#                                                                            #
# Linear Path Optimization                                                   #
#                                                                            #
###############################################################################

# Function to calculate the angle between two points in degrees
# Read the GPS data from the CSV file
data = pd.read_csv('/home/home/Desktop/Projects/pawpatrol/data/gpsdata.csv', header=None, names=['timestamp', 'latitude', 'longitude'])

# Count the total number of points
total_points = len(data)

# Function to calculate the angle between two points in degrees
def calculate_angle(p1, p2):
    delta_y = p2[0] - p1[0]
    delta_x = p2[1] - p1[1]
    angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)  # Convert radians to degrees
    return angle

# Initialize lists to hold filtered points for the two passes
filtered_latitudes = []
filtered_longitudes = []

# First pass: Horizontal optimization
threshold_angle = 5  # You can adjust this threshold

# Collect the first point
filtered_latitudes.append(data['latitude'].iloc[0])
filtered_longitudes.append(data['longitude'].iloc[0])

# Iterate through the data points
for i in range(1, total_points):
    prev_point = (data['latitude'].iloc[i-1], data['longitude'].iloc[i-1])
    curr_point = (data['latitude'].iloc[i], data['longitude'].iloc[i])
    
    # Calculate the angle between the previous and current point
    angle = calculate_angle(prev_point, curr_point)
    
    # If the angle deviation exceeds the threshold (indicating a significant horizontal change), collect the current point
    if abs(angle) > threshold_angle:
        filtered_latitudes.append(curr_point[0])
        filtered_longitudes.append(curr_point[1])

# Second pass: Vertical optimization
optimized_latitudes = []
optimized_longitudes = []

# Initialize lists to hold the points for the vertical optimization
if filtered_latitudes:
    optimized_latitudes.append(filtered_latitudes[0])
    optimized_longitudes.append(filtered_longitudes[0])

# Threshold for vertical pass
min_distance = 1.0  # Minimum distance in meters

# Iterate through the filtered data points
for i in range(1, len(filtered_latitudes)):
    prev_point = (filtered_latitudes[i-1], filtered_longitudes[i-1])
    curr_point = (filtered_latitudes[i], filtered_longitudes[i])
    
    # Calculate the distance between the previous and current point
    distance = haversine(prev_point, curr_point)
    
    # If the distance exceeds the minimum distance, collect the current point
    if distance > min_distance:
        optimized_latitudes.append(curr_point[0])
        optimized_longitudes.append(curr_point[1])

# Plotting the optimized path
plt.figure(figsize=(10, 6))
plt.plot(optimized_longitudes, optimized_latitudes, color='red', linestyle='-', linewidth=2)  # Red line
plt.scatter(optimized_longitudes, optimized_latitudes, color='blue', marker='o')  # Blue dots
plt.title('Optimized GPS Coordinates Path')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.axis('equal')  # Equal aspect ratio ensures the plot is not distorted

# Show total optimized points on the graph
plt.text(0.05, 0.95, f'Total Optimized Points: {len(optimized_latitudes)}', transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='lightgray'))

plt.show()

# Print the total number of optimized points in the console
print(f'Total number of optimized GPS points: {len(optimized_latitudes)}')


###############################################################################
#                                                                            #
# Threshold Bubble Optimization                                              #
#                                                                            #
###############################################################################
# Haversine formula to calculate distance between two GPS points
def haversine(coord1, coord2):
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # Distance in meters
    return distance

# Read the GPS data from the CSV file
data = pd.read_csv('gpsdata.csv', header=None, names=['timestamp', 'latitude', 'longitude'])

# Count the total number of points
total_points = len(data)

# Initialize lists to hold filtered points
filtered_latitudes = []
filtered_longitudes = []

# Thresholds
threshold_angle = 5  # Angle deviation threshold in degrees
min_distance = 1.0  # Minimum distance in meters

# Collect the first point
filtered_latitudes.append(data['latitude'].iloc[0])
filtered_longitudes.append(data['longitude'].iloc[0])

# Iterate through the data points
for i in range(1, total_points):
    prev_point = (data['latitude'].iloc[i-1], data['longitude'].iloc[i-1])
    curr_point = (data['latitude'].iloc[i], data['longitude'].iloc[i])
    
    # Calculate the angle between the previous and current point
    angle = calculate_angle(prev_point, curr_point)
    
    # Calculate the distance between the previous and current point
    distance = haversine(prev_point, curr_point)
    
    # If the angle deviation exceeds the threshold or the distance exceeds the minimum distance, collect the current point
    if abs(angle) > threshold_angle or distance > min_distance:
        filtered_latitudes.append(curr_point[0])
        filtered_longitudes.append(curr_point[1])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(filtered_longitudes, filtered_latitudes, color='red', linestyle='-', linewidth=2)  # Red line
plt.scatter(filtered_longitudes, filtered_latitudes, color='blue', marker='o')  # Blue dots
plt.title('Filtered GPS Coordinates Path')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.axis('equal')  # Equal aspect ratio ensures the plot is not distorted

# Show total filtered points on the graph
plt.text(0.05, 0.95, f'Total Filtered Points: {len(filtered_latitudes)}', transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='lightgray'))

plt.show()

# Print the total number of filtered points in the console
print(f'Total number of filtered GPS points: {len(filtered_latitudes)}')

###############################################################################
# OPT 1 and then OPT 2 #
###############################################################################

# Read the GPS data from the CSV file
data = pd.read_csv('/home/home/Desktop/Projects/pawpatrol/data/gpsdata.csv', header=None, names=['timestamp', 'latitude', 'longitude'])

# Count the total number of points
total_points = len(data)

# Function to calculate the angle between two points in degrees
def calculate_angle(p1, p2):
    delta_y = p2[0] - p1[0]
    delta_x = p2[1] - p1[1]
    angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)  # Convert radians to degrees
    return angle

# Linear Path Optimization
# Initialize lists to hold filtered points
filtered_latitudes = []
filtered_longitudes = []

# Threshold angle in degrees for linear optimization
threshold_angle = 5  # You can adjust this threshold

# Collect the first point
filtered_latitudes.append(data['latitude'].iloc[0])
filtered_longitudes.append(data['longitude'].iloc[0])

# Iterate through the data points
for i in range(1, total_points):
    prev_point = (data['latitude'].iloc[i-1], data['longitude'].iloc[i-1])
    curr_point = (data['latitude'].iloc[i], data['longitude'].iloc[i])
    
    # Calculate the angle between the previous and current point
    angle = calculate_angle(prev_point, curr_point)
    
    # If the angle deviation exceeds the threshold, collect the current point
    if abs(angle) > threshold_angle:
        filtered_latitudes.append(curr_point[0])
        filtered_longitudes.append(curr_point[1])

# Now apply the second optimization on the filtered points
optimized_latitudes = []
optimized_longitudes = []

# Haversine formula to calculate distance between two GPS points
def haversine(coord1, coord2):
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # Distance in meters
    return distance

# Initialize lists to hold the points for the second optimization
if filtered_latitudes:
    optimized_latitudes.append(filtered_latitudes[0])
    optimized_longitudes.append(filtered_longitudes[0])

# Thresholds for the second optimization
threshold_angle_2 = 5  # Angle deviation threshold in degrees
min_distance = 1.0  # Minimum distance in meters

# Iterate through the filtered data points
for i in range(1, len(filtered_latitudes)):
    prev_point = (filtered_latitudes[i-1], filtered_longitudes[i-1])
    curr_point = (filtered_latitudes[i], filtered_longitudes[i])
    
    # Calculate the angle between the previous and current point
    angle = calculate_angle(prev_point, curr_point)
    
    # Calculate the distance between the previous and current point
    distance = haversine(prev_point, curr_point)
    
    # If the angle deviation exceeds the threshold or the distance exceeds the minimum distance, collect the current point
    if abs(angle) > threshold_angle_2 or distance > min_distance:
        optimized_latitudes.append(curr_point[0])
        optimized_longitudes.append(curr_point[1])

# Plotting the optimized path
plt.figure(figsize=(10, 6))
plt.plot(optimized_longitudes, optimized_latitudes, color='red', linestyle='-', linewidth=2)  # Red line
plt.scatter(optimized_longitudes, optimized_latitudes, color='blue', marker='o')  # Blue dots
plt.title('Optimized GPS Coordinates Path')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.axis('equal')  # Equal aspect ratio ensures the plot is not distorted

# Show total optimized points on the graph
plt.text(0.05, 0.95, f'Total Optimized Points: {len(optimized_latitudes)}', transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='none', facecolor='lightgray'))

plt.show()

# Print the total number of optimized points in the console
print(f'Total number of optimized GPS points: {len(optimized_latitudes)}')

###############################################################################
###############################################################################
