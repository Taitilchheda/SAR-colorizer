import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

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

def load_and_preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(128, 128))
    img_array = img_to_array(img)
    img_array_rgb = np.repeat(img_array, 3, axis=-1)
    img_array_rgb = (img_array_rgb / 127.5) - 1
    img_array_rgb = np.expand_dims(img_array_rgb, axis=0)
    return img_array, img_array_rgb

def colorize_image(model, image_path):
    grayscale_image, input_image = load_and_preprocess_image(image_path)
    colorized_image = model.predict(input_image)[0]
    colorized_image = (colorized_image + 1) / 2.0
    return grayscale_image, colorized_image

def save_colorized_image(grayscale_image, colorized_image, output_path):
    colorized_image = (colorized_image * 255).astype(np.uint8)
    colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, colorized_image)

def colorize_and_save(input_path, output_path):
    model = mod_Unet()
    model.load_weights("D:\dj sanghvi\sem 3\SIH 2024\colorizer\gen0.h5")
    grayscale_image, colorized_image = colorize_image(model, input_path)
    save_colorized_image(grayscale_image, colorized_image, output_path)
