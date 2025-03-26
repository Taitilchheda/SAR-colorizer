from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import time
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload and output directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'colorized'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load model only once during initialization
model = None

# Custom Instance Normalization Layer
class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer='ones',
                                     trainable=True)
        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs, training=True):
        mean, variance = tf.nn.moments(inputs, [1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

def d_block(x_input, filters, strides, padding, batch_norm, inst_norm):
    x = tf.keras.layers.Conv2D(filters, (4, 4), strides=strides, padding=padding, use_bias=False, kernel_initializer='random_normal')(x_input)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    if inst_norm:
        x = InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    return x

def u_block(x, skip, filters, strides, padding, batch_norm, inst_norm):
    x = tf.keras.layers.Conv2DTranspose(filters, (4, 4), strides=strides, padding=padding, use_bias=False, kernel_initializer='random_normal')(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    if inst_norm:
        x = InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    return x

def mod_Unet():
    srcI = tf.keras.Input(shape=(128, 128, 3))

    # Contracting path
    c064 = d_block(srcI, 64, 2, 'same', False, False)
    c128 = d_block(c064, 128, 2, 'same', True, False)
    c256 = d_block(c128, 256, 2, 'same', True, False)
    c512 = d_block(c256, 512, 2, 'same', True, False)
    d512 = d_block(c512, 512, 2, 'same', True, False)
    e512 = d_block(d512, 512, 2, 'same', True, False)

    # Bottleneck layer
    f512 = d_block(e512, 512, 2, 'same', True, False)

    # Expanding path
    u512 = u_block(f512, e512, 512, 2, 'same', True, False)
    u512 = u_block(u512, d512, 512, 2, 'same', True, False)
    u512 = u_block(u512, c512, 512, 2, 'same', True, False)
    u256 = u_block(u512, c256, 256, 2, 'same', True, False)
    u128 = u_block(u256, c128, 128, 2, 'same', True, False)
    u064 = u_block(u128, c064, 64, 2, 'same', False, True)

    genI = tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='tanh', kernel_initializer='random_normal')(u064)

    model = tf.keras.Model(inputs=srcI, outputs=genI)
    return model

def load_model():
    global model
    if model is None:
        model = mod_Unet()
        # Update this path to a relative or environment variable path
        model_path = os.environ.get('MODEL_PATH', 'E:\SAR Colorizer\SAR-Colorizer\gen0.h5')
        model.load_weights(model_path)
        print("Model loaded successfully")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(128, 128))
    img_array = img_to_array(img)
    img_array_rgb = np.repeat(img_array, 3, axis=-1)
    img_array_rgb = (img_array_rgb / 127.5) - 1
    img_array_rgb = np.expand_dims(img_array_rgb, axis=0)
    return img_array, img_array_rgb

def colorize_image(image_path):
    _, input_image = load_and_preprocess_image(image_path)
    colorized_image = model.predict(input_image)[0]
    colorized_image = (colorized_image + 1) / 2.0
    colorized_image = (colorized_image * 255).astype(np.uint8)
    return colorized_image

def save_colorized_image(colorized_image, output_path):
    colorized_image_bgr = cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, colorized_image_bgr)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        input_filename = f"{unique_id}.{file_extension}"
        output_filename = f"{unique_id}_colorized.{file_extension}"
        
        # Save paths
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Save uploaded file
        file.save(input_path)
        
        # Ensure model is loaded
        load_model()
        
        # Process and colorize image
        try:
            start_time = time.time()
            colorized_image = colorize_image(input_path)
            save_colorized_image(colorized_image, output_path)
            processing_time = time.time() - start_time
            
            return jsonify({
                'success': True,
                'original': input_filename,
                'colorized': output_filename,
                'processing_time': processing_time
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/images/<path:filename>')
def get_image(filename):
    # Determine which folder to serve from based on whether "colorized" is in the filename
    if 'colorized' in filename:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Load model on startup
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)