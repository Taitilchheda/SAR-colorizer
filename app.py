from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from model import colorize_and_save

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "D:/dj sanghvi/sem 3/SIH 2024/colorizer/static/uploads"
RESULT_FOLDER = "D:/dj sanghvi/sem 3/SIH 2024/colorizer/static/results"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)

        file.save(input_path)
        
        # Process the image and save the colorized result
        colorize_and_save(input_path, result_path)
        
        # Return URLs for the images
        return jsonify({
            'original_url': f'/static/uploads/{filename}',
            'colorized_url': f'/static/results/{filename}'
        })

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
