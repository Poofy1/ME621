from flask import Flask, render_template, request, jsonify
from e621_backend.annotation import annotation_bp
from model.routes import training_bp
import webbrowser, os, json
from config import global_config, initialize_global_config, save_config, load_config
import threading
import queue
import tqdm
import requests
current_dir = os.path.dirname(os.path.abspath(__file__))

# Register blueprints
app = Flask(__name__)
app.register_blueprint(annotation_bp, url_prefix='/annotation')
app.register_blueprint(training_bp, url_prefix='/training')



# Queue for communication between threads
task_queue = queue.Queue()

# Global variable to store training status
training_status = {"status": "idle"}


@app.route('/')
def index():
    return render_template('index.html', config_exists=global_config['config'] is not None)

@app.route('/save_config', methods=['POST'])
def save_config_route():
    username = request.form['username']
    api_key = request.form['api_key']
    chat_id = request.form['chat_id']
    bot_api = request.form['bot_api']
    save_config(username, api_key, chat_id, bot_api)
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
            training_status["status"] = "running"
            train_model()
            training_status["status"] = "completed"
        task_queue.task_done()




def download_file(url, filepath):
    # Check if file already exists
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Send a GET request to the URL
    print(f"Downloading {os.path.basename(filepath)}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Open the file and write the content
    with open(filepath, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    
    print(f"File downloaded successfully: {filepath}")

    
    
def launch_me621():
    # Download model if it does not exist
    model_path = os.path.join(current_dir, "load_model", "models", "Stable-diffusion", "yiffymix_v44.safetensors")
    model_url = "https://civitai.com/api/download/models/558148?type=Model&format=SafeTensor&size=full&fp=fp16"
    download_file(model_url, model_path)
    
    config_path = os.path.join(current_dir, "load_model", "models", "Stable-diffusion", "yiffymix_v44.yaml")
    config_url = "https://civitai.com/api/download/models/558148?type=Config&format=Other"
    download_file(config_url, config_path)


    # Initialize global configuration when the app starts
    load_config()
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Open the browser
    webbrowser.open_new('http://127.0.0.1:5000/')

    # Run main thread tasks
    main_thread_tasks()