from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from typing import Union

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


if __name__ == '__main__':
    app.run(debug=True)
