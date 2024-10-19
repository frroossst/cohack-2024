import pandas as pd

def compress_csv(input_df):
    count = 0

    # Work directly with the DataFrame
    df = input_df

    # Create a list to store the compressed data
    compressed_data = []

    # Initialize the first row's latitude and longitude
    longitude_first = df.iloc[0]['Longitude']
    latitude_first = df.iloc[0]['Latitude']

    # Append the first row as it is, because it's always included
    compressed_data.append([df.iloc[0]['Time'], latitude_first, longitude_first])

    # Iterate through each row starting from the second row
    for index, row in df.iterrows():
        if index == 0:
            continue  # Skip the first row since it's already added

        time = row['Time']
        latitude = row['Latitude']
        longitude = row['Longitude']

        # Calculate the differences in latitude and longitude
        delta_y = abs(longitude - longitude_first)
        delta_x = abs(latitude - latitude_first)

        if count == 0:
            count_lat = latitude
            count_long = longitude

        # Only update if both latitude and longitude have changed significantly
        if (delta_y > 0.00004 and delta_x > 0.00003) or (delta_y > 0.00003 and delta_x > 0.00004):
            compressed_data.append([time, latitude, longitude])
            # Update the reference points for the next comparison
            count += 1
            longitude_first = longitude
            latitude_first = latitude
        if count == 5:
            delta_count_y = abs(longitude - count_long)
            delta_count_x = abs(latitude - count_lat)
            if delta_count_y > 0.00004 and delta_count_x > 0.00004:
                compressed_data.pop()
                compressed_data.pop()
                compressed_data.pop()
            count = 0

    # Convert the compressed data to a DataFrame
    # compressed_df = pd.DataFrame(compressed_data, columns=['Time', 'Latitude', 'Longitude'])
    # Convert the compressed data to a DataFrame
    compressed_df = pd.DataFrame(compressed_data)
    compressed_df.columns = ['timestamp', 'latitude', 'longitude']
    compressed_df.header = None

    # Convert DataFrame to a list of dictionaries (orient='records')
    compressed_dict = compressed_df.to_dict(orient='records')

    return compressed_dict

if __name__ == "__main__":
    # Load the CSV file into a DataFrame
    df = pd.read_csv('/home/home/Desktop/Projects/pawpatrol/data/cutesiedata.csv', header=None, names=['Time', 'Latitude', 'Longitude'])

    # Get the length of the data
    len_data = len(df)

    # Colton's optimisation
    result_df = compress_csv(df)

    # Check that the result has fewer rows than the original
    assert len(result_df) < len_data
