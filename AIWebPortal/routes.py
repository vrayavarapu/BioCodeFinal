import os
from flask import render_template, send_from_directory, jsonify
from app import app

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/inference')
def inference():
    return render_template('inference.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/model/<path:filename>')
def serve_model(filename):
    # Ensure the model directory exists relative to the app
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        app.logger.warning(f"Created model directory at: {model_dir}")

    # Check if the requested file exists
    file_path = os.path.join(model_dir, filename)
    if not os.path.exists(file_path):
        app.logger.error(f"Model file not found: {filename}")
        app.logger.error(f"Expected path: {file_path}")
        return jsonify({"error": f"Model file not found: {filename}", "path": file_path}), 404

    app.logger.info(f"Serving model file: {filename} from {file_path}")
    return send_from_directory(model_dir, filename)