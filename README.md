### Food Classification Neural Network for the first round of the Oracle challenge
This is a project that involves building a neural network to classify 8 different classes of foods using Python and the TensorFlow library. The goal of the project is to create a machine learning model that can accurately identify different types of foods based on the images provided for the competition.

### Dataset
The dataset used for this project is a collection of images of foods provided by Nuwe.


The dataset contains around 18.000 images in total, if we account for the train and test. With each class having roughly 1,250 images in the training dataset. The dataset is split into a training set (80%) and a validation set (20%) from the original train dataset. Next, the predictions are made on the test dataset.

### Neural Network Architecture
The neural network architecture used for this project is a convolutional neural network (CNN). The CNN is composed of multiple convolutional layers, followed by pooling layers and a fully connected layer at the final block. The final layer is a softmax layer that produces a probability distribution over the 8 classes. The neural network is built of 6 different blocks. 

### Usage
To run the code, simply run the "main.py" file. This will train the neural network using the training set and evaluate its performance using the validation set. The trained model will then be saved to a file called "model_weights.h5".

### Dependencies
All the dependencies and libraries can be found in the file "requirements.txt".

### Acknowledgements
This project was completed as part of the Oracle League. The dataset used in this project was provided by Nuwe, the rules stated that no transfer learning nor outsourcing data was allowed. Thus, all the development was made with the given resources and the neural network was build from scratch.
