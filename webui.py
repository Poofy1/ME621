from flask import Flask, render_template
from e621_backend.annotation import *

import webbrowser
from threading import Timer

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from model.routes import *

app = Flask(__name__)

# Register blueprints
app.register_blueprint(annotation_bp, url_prefix='/annotation')
app.register_blueprint(training_bp, url_prefix='/training')

@app.route('/')
def index():
    return render_template('index.html')

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True, use_reloader=False)