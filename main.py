from flask import Flask, render_template, jsonify, redirect, url_for
import pandas as pd

from data import concatenate, opti

app = Flask(__name__)


def log(message: str) -> None:
    filepath = '/home/home/Desktop/Projects/pawpatrol/log.txt'
    with open(filepath, 'a') as log_file:
        log_file.write('[DEBUG] '+ message + '\n')

def clear_log() -> None:
    filepath = '/home/home/Desktop/Projects/pawpatrol/log.txt'
    with open(filepath, 'w') as log_file:
        log_file.write('')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')  # Render the blank dashboard page

# Route to serve the login page (GET request)
@app.route('/login', methods=['GET'])
def login_page() -> str:
    return render_template('login.html')

@app.route('/')
def hello():
    return redirect(url_for('login_page'))

@app.route('/data', methods=['GET'])
def get_gps_data():
    try:
        df = pd.read_csv('/home/home/Desktop/Projects/pawpatrol/data/cutesiedata.csv', header=None, names=['Time', 'Latitude', 'Longitude'])

        # Get the length of the data
        len_data = len(df)

        # Colton's optimisation
        result_df = opti.compress_csv(df)

        assert len(result_df) <= len_data, f"Expected compressed data to be smaller than original data, but got {len(result_df)} > {len_data}"

        # Return the points as a JSON response
        return jsonify(result_df)
    except Exception as e:
        log(str(e))
        return jsonify({"error": str(e)}), 500 

@app.route('/activity', methods=['GET'])
def get_activity_data():
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv('/home/home/Desktop/Projects/pawpatrol/data/activity_points.csv', header=None, names=['timestamp', 'latitude', 'longitude'])

        result_df = concatenate.conc(df, threshold=0.0009)

        # Convert the DataFrame to a list of dictionaries
        points = result_df.to_dict(orient='records')

        # Return the points as a JSON response
        return jsonify(points)
    except Exception as e:
        return jsonify({"error": str(e)}), 500 

@app.route('/statistics', methods=['GET'])
def get_statistics():
    df = pd.read_csv('/home/home/Desktop/Projects/pawpatrol/data/activity_points.csv', header=None, names=['timestamp', 'latitude', 'longitude'])
    result_df = concatenate.conc(df, threshold=0.0009)
    points = result_df.to_dict(orient='records')

    return jsonify({})

@app.route('/version', methods=['GET'])
def get_version():
    # return html string showig a centered version number
    return "<h4 style='text-align:center;'>v0.28.137</h4>"

if __name__ == '__main__':
    app.run(debug=True)
