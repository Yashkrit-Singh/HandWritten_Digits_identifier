# Handwritten Digits Identifier

This project aims to build a machine learning model that can classify handwritten digits using the MNIST dataset. The model uses a Neural Network (NN) built with TensorFlow and Keras to recognize digits from 0 to 9.

## Project Overview

The Handmade Digits Identifier uses a machine learning model to process handwritten digits. The model is trained on the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits.

The key steps of the project are:
1. **Loading and Preprocessing the Data**: The MNIST dataset is loaded, pixel values are normalized, and data is reshaped to be suitable for neural network training.
2. **Building the Model**: The model is designed using a Sequential approach with Dense layers and ReLU activations.
3. **Training the Model**: The model is trained using 80% of the data, with 20% held out for validation. The training process involves multiple epochs and mini-batches.
4. **Evaluating the Model**: The trained model is tested on unseen data (test set) to evaluate its accuracy.
5. **Making Predictions**: Sample predictions are displayed to visualize the model’s performance on test data.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [How to Run](#how-to-run)
- [License](#license)

## Technologies Used

- **TensorFlow**: For building and training the model.
- **Keras**: A high-level neural networks API, running on top of TensorFlow.
- **Matplotlib**: For visualizing data and model predictions.
- **NumPy**: For data manipulation and numerical operations.
- **Python**: Programming language used for development.

## Dataset

The model is trained on the **MNIST dataset**, a collection of 60,000 labeled 28x28 pixel grayscale images of handwritten digits (0-9). The dataset also includes a test set of 10,000 images.

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Input Size**: 28x28 pixels, grayscale images
- **Output Classes**: 10 (digits from 0 to 9)

## Model Architecture

The model is built using a **Sequential model** with the following layers:
1. **Input Layer**: The 28x28 input images are flattened into a vector of size 784 (28 * 28).
2. **Hidden Layer**: A fully connected layer with 128 units and ReLU activation to introduce non-linearity.
3. **Output Layer**: A fully connected layer with 10 units (one for each digit) and softmax activation to predict the probability of each digit class.

## Training

The model is trained using the following parameters:
- **Epochs**: The model is trained for 10 epochs, meaning it will pass through the entire dataset 10 times.
- **Batch Size**: During each epoch, the data is processed in batches of 32 samples before updating the model's weights.
- **Validation Split**: 20% of the data is used for validation to evaluate the model during training.

## How to Run

1. Install the necessary dependencies by running the following:
   ```bash
   pip install tensorflow matplotlib numpy
