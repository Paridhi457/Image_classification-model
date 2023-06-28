# Image_classification-model using CNN
This repository contains code for building an image classification model using Convolutional Neural Networks (CNN) with TensorFlow. The model is trained to classify images into two classes: "Sad" and "Happy". This README file provides an overview of the code and concepts used in the project.

Dependencies:
TensorFlow
OpenCV (cv2)
NumPy
Matplotlib

Data Preparation:
The image dataset is expected to be located in a directory called "data".
Images should be organized into subdirectories, where each subdirectory represents a different class.

Data Preprocessing:
The dataset is loaded using TensorFlow's image_dataset_from_directory function.
The dataset is divided into training, validation, and test sets.
The training set comprises 70% of the data, the validation set has 20%, and the test set has 10%.
The pixel values of the images are scaled to the range [0, 1].

Model Architecture:
The model architecture consists of a series of convolutional and pooling layers followed by fully connected layers.
The convolutional layers learn features from the input images, and the pooling layers reduce spatial dimensions.
The final fully connected layers classify the learned features into the two classes.
The model uses ReLU activation for convolutional layers and sigmoid activation for the final output layer.
The model is compiled with the Adam optimizer and Binary Crossentropy loss function.

Model Training:
The model is trained using the training set and validated on the validation set.
The training process is monitored using the TensorBoard callback, which logs metrics and visualizes them.
The model is trained for 20 epochs

Evaluation:
The trained model is evaluated on the test set using metrics such as precision, recall, and binary accuracy.
The evaluation results are printed to the console.

Inference:
An example image ("154006829.jpg") is loaded using OpenCV (cv2) and displayed using Matplotlib.
The image is preprocessed by scaling its pixel values and resizing it to the required input size.
The preprocessed image is passed through the trained model for inference.
The predicted class is determined based on the output probability. If the probability is greater than 0.5, the class is predicted as "Sad"; otherwise, it is predicted as "Happy".
