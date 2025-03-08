# SAR Image Colorizer - GAN-based Colorization of SAR Images

## Overview

The **SAR Image Colorizer** is an advanced machine learning project that utilizes **Generative Adversarial Networks (GANs)** to colorize **Synthetic Aperture Radar (SAR)** images. This deep learning model transforms grayscale SAR images into high-quality colorized versions, making it a powerful tool for enhancing satellite and aerial imagery.

The project leverages state-of-the-art techniques, including a **U-Net generator**, **PatchGAN discriminator**, and a **multi-scale approach** to produce visually realistic and detailed results. It’s designed to work with large-scale SAR datasets and can be deployed to generate color images from grayscale inputs with minimal artifacts.

## Key Features

- **GAN-based Architecture**: Uses a **U-Net** generator and **PatchGAN** discriminator to generate high-quality colorized SAR images.
- **Multi-scale Approach**: The model employs a multi-scale discriminator (with PatchGAN at different scales) to improve the image's realism and prevent overfitting.
- **Loss Functions**: Combines **adversarial loss** with **L1 loss** to ensure both perceptual quality and pixel-wise accuracy.
- **Normalization Layers**: Batch and Instance Normalization layers to stabilize training and enhance generalization across different SAR image data.
- **Efficient Data Pipeline**: Utilizes TensorFlow’s `tf.data.Dataset` API for handling large-scale SAR datasets efficiently, including image resizing, normalization, and augmentation.

## Technologies Used

- **TensorFlow** & **Keras**: The framework used for developing and training the deep learning models.
- **Generative Adversarial Networks (GANs)**: The core technique used for generating colorized images from grayscale SAR data.
- **U-Net**: Custom deep learning architecture used as the generator to convert grayscale SAR images to color.
- **PatchGAN**: A discriminator model that performs local image discrimination for better quality.
- **Adam Optimizer**: Optimizer used for training the models with learning rate tuning.
- **Batch Normalization & Instance Normalization**: Techniques used to stabilize and speed up training.
- **OpenCV**: Used for preprocessing and image manipulation, such as resizing and color conversion.
- **Matplotlib**: For data visualization and displaying results.

## Installation

To run the SAR Image Colorizer, you'll need the following dependencies:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sar-image-colorizer.git
   cd sar-image-colorizer
