from flask import Flask, render_template, request, jsonify
from e621_backend.annotation import annotation_bp
from model.routes import training_bp
import webbrowser, os, json
from config import global_config, initialize_global_config
import threading
import queue

project_root = os.path.abspath(os.path.dirname(__file__))

# Register blueprints
app = Flask(__name__)
app.register_blueprint(annotation_bp, url_prefix='/annotation')
app.register_blueprint(training_bp, url_prefix='/training')

CONFIG_FILE = f'{project_root}/config.json'

# Queue for communication between threads
task_queue = queue.Queue()

# Global variable to store training status
training_status = {"status": "idle"}

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

@app.route('/start_training', methods=['POST'])
def start_training():
    task_queue.put('train')
    return jsonify({"status": "Training started"})

@app.route('/training_status', methods=['GET'])
def get_training_status():
    return jsonify(training_status)

def run_flask():
    app.run(debug=False, use_reloader=False)

def main_thread_tasks():
    global training_status
    while True:
        task = task_queue.get()  # This will block until an item is available
        if task == 'train': # I know this is shit but I could not find another way
            from model.trainer import train_model
            print("Starting training on main thread...")
            training_status["status"] = "running"
            train_model()
            training_status["status"] = "completed"
            print("Training completed.")
        task_queue.task_done()

if __name__ == "__main__":
    # Initialize global configuration when the app starts
    initialize_global_config(load_config())
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Open the browser
    webbrowser.open_new('http://127.0.0.1:5000/')

    # Run main thread tasks
    main_thread_tasks()