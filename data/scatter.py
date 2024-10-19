import csv
import matplotlib.pyplot as plt

# Read data from gpsdata.csv
latitudes = []
longitudes = []

with open('gpsdata.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        latitudes.append(float(row[1]))
        longitudes.append(float(row[2]))

# Create a scatter plot
plt.scatter(longitudes, latitudes, color='blue')

# Draw a line through each point
plt.plot(longitudes, latitudes, color='red')

# Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('GPS Data Points')

# Show the plot
plt.show()
