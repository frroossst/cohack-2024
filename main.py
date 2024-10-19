from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from typing import Union
import pandas as pd

from data import concatenate

app = Flask(__name__)


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
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv('/home/home/Desktop/Projects/pawpatrol/data/cutesiedata.csv', header=None, names=['timestamp', 'latitude', 'longitude'])

        # Convert the DataFrame to a list of dictionaries
        points = df.to_dict(orient='records')
        
        # Return the points as a JSON response
        return jsonify(points)
    except Exception as e:
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


if __name__ == '__main__':
    app.run(debug=True)
