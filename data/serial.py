import serial
import time
import sys
import serial.tools.list_ports


def read_serial(port, baudrate, output_file):
    try:
        # Open the serial port
        with serial.Serial(port, baudrate, timeout=1) as ser, open(output_file, 'wb') as file:
            print(f"Reading from {port} at {baudrate} baud...")
            while True:
                # Read a line of data from the serial port
                data = ser.read_until()  # You can adjust this to read a specific number of bytes or use ser.readline()

                if data:
                    # Write the data to the file
                    file.write(data)
                    file.flush()  # Ensure data is written immediately
                    print(data)  # Optional: print data to the console for monitoring

                time.sleep(0.1)  # Optional: add a small delay to avoid high CPU usage

    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")


def find_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


if __name__ == "__main__":
    # Find available COM ports
    available_ports = find_serial_ports()

    if not available_ports:
        print("No serial ports found.")
        sys.exit(1)

    # Use the first available port
    serial_port = available_ports[0]
    baud_rate = 115200  # Change to your desired baud rate
    output_file_name = 'output.txt'  # Change to your desired output file name

    print(f"Using serial port: {serial_port}")
    read_serial(serial_port, baud_rate, output_file_name)

