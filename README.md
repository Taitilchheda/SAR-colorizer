# 🌈 SAR Image Colorizer

![SAR Image Colorizer](https://github.com/Taitilchheda/SAR-colorizer/blob/main/sar%20img%203%20phase.jpg)

This project is aimed at developing a **Generative Adversarial Network (GAN)** based system for colorizing **grayscale Synthetic Aperture Radar (SAR) images**. The model utilizes deep learning techniques such as **U-Net** for the generator and **PatchGAN** for the discriminator to produce high-quality, realistic colorized SAR images. 🖼️

## 📜 Overview

SAR image colorization is important in improving the interpretability of radar images, which are typically monochromatic and lack detailed features. This project uses a GAN-based approach to convert grayscale SAR images into colorized images, enabling better visualization and analysis of radar data. 🎨

## 🔧 Features

- **Generative Adversarial Network (GAN)** based colorization using **U-Net** (generator) and **PatchGAN** (discriminator). 🤖
- Combines **adversarial loss** and **L1 loss** for enhanced image quality and accuracy. 🏆
- Implements **Batch Normalization** and **Instance Normalization** for stable and efficient training. 📈
- Supports real-time training and evaluation with high-quality visualizations of generated color images. 👨‍💻
- Built using **TensorFlow** and **Keras** for deep learning. 🔥

## 💻 Easy User Interface
The project also provides a user-friendly interface that simplifies the process of colorizing grayscale SAR images. With just a few clicks, users can easily upload their grayscale images, and the model will generate the colorized output in real-time.
![SAR Image Colorizer UI](https://github.com/Taitilchheda/SAR-colorizer/blob/main/UI%20for%20SAR.jpg).

Key features of the UI:

- Drag-and-drop support for uploading images. 🖱️
- Preview mode to check the results before downloading the colorized images. 🔍
- Batch processing to colorize multiple images at once. ⚙️
- Intuitive navigation and simple controls to start and stop the process. 🚀

## 🛠️ Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sar-image-colorizer.git
   cd sar-image-colorizer

2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Prepare your SAR dataset with grayscale and color images. The folder structure should be:
    ```bash
    /path/to/dataset/
        ├── gray_images/
        └── color_images/
    ```
4. Train the model:
    ```bash
    python train.py
    ```
5. To evaluate the model and generate colorized images:
    ```bash
    python evaluate.py --input_path /path/to/test_images/ --output_path /path/to/save_colorized_images/
    ```

## 💡 Example

Here’s an example of how the model colorizes grayscale SAR images:

| Input Grayscale Image | Generated Color Image |
|-----------------------|-----------------------|
| ![Input Image](https://github.com/Taitilchheda/SAR-colorizer/blob/main/grey%20image.png) | ![Generated Image](https://github.com/Taitilchheda/SAR-colorizer/blob/main/colored%20image.png) |

## 🤝 Acknowledgements

This project is inspired by U-Net and GAN architectures commonly used for image-to-image tasks like segmentation and colorization.

Special thanks to the TensorFlow and Keras teams for providing excellent deep learning libraries.

The dataset used for training the model comes from Sentinel-1&2 Image Pairs (SAR & Optical) Kaggle.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For any questions or suggestions, feel free to open an issue or contact me at taitilchheda@gmail.com.
