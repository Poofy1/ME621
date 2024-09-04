import logging
from flask import Flask, render_template, request, jsonify, Response
from e621_backend.annotation import annotation_bp
import webbrowser, os
from config import global_config, save_config, load_config
import threading
import queue
from datetime import datetime
import requests
import io
import csv
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))


class SuppressedLogFilter(logging.Filter):
    def filter(self, record):
        return not (
            ('GET /console_output' in record.getMessage()) or
            ('GET /training_status' in record.getMessage())
        )
        
        
app = Flask(__name__)
app.register_blueprint(annotation_bp, url_prefix='/annotation')
logging.getLogger('werkzeug').addFilter(SuppressedLogFilter())

task_queue = queue.Queue()
training_status = {"status": "idle"}
console_output = io.StringIO()


def get_model_stats():
    csv_path = f'{current_dir}/data/models/model_stats.csv'
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row = next(reader)
            return {
                'datetime': datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'),
                'accuracy': float(row[1]),
                'images_trained': int(row[2])
            }
    return None

@app.route('/')
def index():
    model_stats = get_model_stats()
    return render_template('index.html', config_exists=global_config['config'] is not None, model_stats=model_stats)

@app.route('/save_config', methods=['POST'])
def save_config_route():
    username = request.form['username']
    api_key = request.form['api_key']
    chat_id = request.form['chat_id']
    bot_api = request.form['bot_api']
    save_config(username, api_key, chat_id, bot_api)
    return jsonify({"success": True})



        
@app.route('/console_output')
def get_console_output():
    return Response(console_output.getvalue(), mimetype='text/plain')

def custom_print(*args, sep=' ', end='\n', file=sys.stdout, flush=False):
    print(*args, sep=sep, end=end, file=file, flush=flush)
    print(*args, sep=sep, end=end, file=console_output, flush=flush)
    
    
@app.route('/training_status', methods=['GET'])
def get_training_status():
    return jsonify(training_status)

@app.route('/start_training', methods=['POST'])
def start_training():
    if training_status["status"] == "idle":
        custom_print("Starting Training...")
        task_queue.put('train')
        training_status["status"] = "running"
        return jsonify({"status": "Training started"})
    else:
        return jsonify({"status": training_status["status"]})

@app.route('/start_bot', methods=['POST'])
def start_bot():
    custom_print("Starting Bot...")
    task_queue.put('bot')
    return jsonify({"status": "Bot started"})
    
def main_thread_tasks():
    global training_status
    from model.trainer import train_model
    from telegram_backend.bot import launch_bot
    while True:
        task = task_queue.get()
        if task == 'train':
            train_model(custom_print)
        elif task == 'bot':
            launch_bot(custom_print)
        training_status["status"] = "idle"
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

    

def run_flask():
    app.run(debug=False, use_reloader=False)


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



if __name__ == "__main__":
    launch_me621()