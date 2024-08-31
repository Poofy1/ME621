from flask import Flask, render_template, request, jsonify
from e621_backend.annotation import annotation_bp
from model.routes import training_bp
import webbrowser
from threading import Timer
import sys
import os
import json
import base64
from config import global_config, initialize_global_config

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
from model.routes import *

app = Flask(__name__)

# Register blueprints
app.register_blueprint(annotation_bp, url_prefix='/annotation')
app.register_blueprint(training_bp, url_prefix='/training')

CONFIG_FILE = f'{project_root}/config.json'

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None

def save_config(username, api_key):
    config = {
        "USERNAME": username,
        "API_KEY": api_key
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Update global configuration
    initialize_global_config(config)

@app.route('/')
def index():
    return render_template('index.html', config_exists=global_config['config'] is not None)

@app.route('/save_config', methods=['POST'])
def save_config_route():
    username = request.form['username']
    api_key = request.form['api_key']
    save_config(username, api_key)
    return jsonify({"success": True})

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    # Initialize global configuration when the app starts
    initialize_global_config(load_config())
    
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)