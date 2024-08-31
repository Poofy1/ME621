from flask import Blueprint, render_template, jsonify, request
import threading
import io
import sys
from contextlib import redirect_stdout
from model.trainer import *

training_bp = Blueprint('training', __name__)

training_output = []
training_thread = None

@training_bp.route('/')
def training_index():
    return render_template('training.html')

@training_bp.route('/start_training', methods=['POST'])
def start_training():
    global training_thread
    if training_thread is None or not training_thread.is_alive():
        training_thread = threading.Thread(target=run_training)
        training_thread.start()
        return jsonify({"status": "Training started"})
    else:
        return jsonify({"status": "Training is already in progress"})

@training_bp.route('/get_output')
def get_output():
    return jsonify({"output": training_output})

def run_training():
    global training_output
    training_output = []
    
    # Redirect stdout to capture print statements
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        train_model()  # Call your training function here
    except Exception as e:
        training_output.append(f"Error occurred: {str(e)}")
    finally:
        sys.stdout = sys.__stdout__
        training_output = captured_output.getvalue().split('\n')