from serial import *
import time

# Set up serial connection
# Change 'COM3' to your port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' or '/dev/ttyACM0' on Linux)
arduino_port = 'COM11'  # Replace with your Arduino port
baud_rate = 115200  # Make sure this matches the baud rate set in your Arduino sketch
output_file = 'arduino_output.txt'  # Name of the file where data will be saved

# Initialize serial connection
ser = Serial(arduino_port, baud_rate)
time.sleep(2)  # Wait for the connection to establish

# Open file to save data
with open(output_file, 'w') as file:
    print("Collecting data... Press Ctrl+C to stop.")
    try:
        while True:
            # Read a line from the serial port
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()  # Read, decode, and remove any extra spaces
                print(line)  # Print the data to the console
                file.write(line + '\n')  # Write the data to a file
    except KeyboardInterrupt:
        print("Data collection stopped.")

# Close the serial connection
ser.close()