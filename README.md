# Fruit-and-Vegetable-Classifier-
Image classification project using MobileNetV2 pretrained on ImageNet for transfer learning. The model classifies images into 36 categories with data augmentation and early stopping. It uses TensorFlow and Keras, and the trained model is saved for prediction on new images.

# Transfer Learning Image Classification with MobileNetV2

## Overview
This project implements an image classification system using transfer learning with the MobileNetV2 architecture pretrained on ImageNet. It classifies images into 36 distinct categories using a custom dataset split into training, validation, and test sets.

MobileNetV2 is chosen for its efficiency and high accuracy with fewer parameters. By freezing the pretrained layers and adding custom dense layers, this project achieves effective classification while reducing training time.

## Features
- Organized dataset with labeled images for training, validation, and testing
- Data augmentation: rotation, zoom, width & height shift, shear, horizontal flip
- Transfer learning with MobileNetV2 and frozen pretrained weights
- Custom densely connected layers for classification
- Early stopping callback to avoid overfitting
- Model saving and loading capability
- Simple prediction function for new images

## Dataset Structure
The dataset folders are structured as:


## Installation
1. Clone the repository:

2. Install required dependencies:


## Usage
- Prepare your dataset in the required folder structure.
- Run training using the provided scripts/notebooks.
- Use the saved model (`FV.h5`) for inference on new images.
- Example prediction code snippet:


## Model Architecture
- Base: MobileNetV2 (without top, ImageNet weights, frozen)
- Two Dense layers with 128 units and ReLU activation
- Output layer with 36 units and softmax activation

## Training Details
- Loss Function: Categorical Cross-Entropy
- Optimizer: Adam
- Batch size: 32
- EarlyStopping monitored on validation loss with patience=2
- Data augmentation applied during training

Feel free to customize the sections (Author, repo URL) before upload. This will provide a professional, clear, and concise README for potential users and contributors.

