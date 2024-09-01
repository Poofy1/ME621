from flask import Blueprint, render_template, jsonify, request
from model.trainer import train_model

training_bp = Blueprint('training', __name__)


@training_bp.route('/')
def training_index():
    return render_template('training.html')

@training_bp.route('/start_training', methods=['POST'])
def start_training():
    
    train_model()
    return jsonify({"status": "Training completed"})
