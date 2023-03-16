import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, MaxPool2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.python import debug as tf_debug

try: 
    import warnings
    warnings.filterwarnings('ignore')
except:
    pass

import analysis as a
from model import MyModel
from callbacks import CustomCallbacks, f1_score


base_dir = "./"
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

a.visualization(base_dir, train, "path_img")

num_classes = train["label"].nunique() # Define the number of classes
target_size = (224, 224) # Define the target_size
batch_size = 64 # Define the batch size
input_shape = (224, 224, 3) # Define the input shape
epochs = 30

# Preprocess
train['label'] = train['label'].apply(lambda x: str(x)) 


train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
total_train = train_df.shape[0]
total_validate = val_df.shape[0]
total_test = test.shape[0]


train_steps = int(total_train / batch_size)
validation_steps = int(total_validate / batch_size)
test_steps = int(total_test / batch_size)


train_datagen = ImageDataGenerator(rescale=1.0/255.,
                                     rotation_range=0.1,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     shear_range=0.1,
                                     zoom_range=0.1,
                                     horizontal_flip=False,
                                     fill_mode="nearest")
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    color_mode="rgb",
    x_col='path_img', # Column containing the image file paths
    y_col='label', # Column containing the image labels
    target_size=target_size, # Size of input images
    batch_size=batch_size,
    class_mode='sparse', # Use categorical cross-entropy loss
    shuffle=True, # Shuffle the order of images
    )

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    color_mode="rgb",
    x_col='path_img',
    y_col='label',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False # Don't shuffle the order of images
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test,
    color_mode="rgb",
    x_col='path_img',
    y_col=None, # No labels for prediction
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None, # Use sparse categorical cross-entropy loss
    shuffle=False,
)


model = MyModel(input_shape=input_shape, num_classes=num_classes)

# Compile the model with the desired metrics and loss function
model.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

# Train the model using the data generators and save the training history
history = model.model.fit(train_generator,
                    epochs=epochs, 
                    validation_data=val_generator,
                    validation_steps=validation_steps,
                    steps_per_epoch=train_steps,
                    # callbacks=[CustomCallbacks()]
                    )

model.model.save_weights("model_weights.h5")
probabilities = model.model.predict(test_generator)

# Get the predicted classes for each image in the test set
predicted_classes = np.argmax(probabilities, axis=-1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predicted_labels = [labels[k] for k in predicted_classes]

# Add the idx_test column from the test dataframe to the results dataframe
results_df = pd.DataFrame({'target': predicted_labels})

# Save the predictions to CSV and JSON files
results_df.to_csv('results.csv')
results_df.to_json('results.json')