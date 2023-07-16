# Handwritten-Digit-Recognition-MNIST-
This project aims to create a model that recognizes handwritten digits using the famous MNIST dataset. The repository contains all the necessary code files and the trained model.

## Project Description
In this project, we train a convolutional neural network (CNN) to classify images of handwritten digits (0-9) from the MNIST dataset. We implement the model using TensorFlow and Keras, and we evaluate different network architectures and hyperparameters to achieve the best accuracy.

## Requirements
* Python 3.6+
* TensorFlow 2.0+
* Keras
* OpenCV (for preprocessing custom images)
* Numpy, Matplotlib (for data visualization)

## Usage
* Clone this repository to your local machine.
* Run the Handwritten_Digit_Recognition_CNN.py file on Google Colab to train the model on the MNIST dataset. This will also save the trained model.
* The test and everything is on the code if you follow the code and read the comment will understand what it does.
* To test the model on your own handwritten digits, preprocess your image by replacing "WhatsApp Image 2023-07-16 at 08.28.28.jpeg" file, and then see the prediction.

## Usage 2
* Clone this repository to your local machine.
* Run the handwritten_digit_recognition_cnn.py file to train the model on the MNIST dataset. This will also save the trained model.
* To test the model on your own handwritten digits, preprocess your image using the image_preprocessing.py file, and then use model_prediction.py to make a prediction.
"""
python model_training.py
python image_preprocessing.py --image your_image.png
python model_prediction.py
"""
* Replace your_image.png with the path to your image.

## Technologies Used
* TensorFlow and Keras: We used these libraries to create and train our neural network.
* OpenCV: We used OpenCV to preprocess custom images to be fed to the trained model for prediction.
* Numpy and Matplotlib: We used Numpy for numerical computations and Matplotlib for data visualization.
