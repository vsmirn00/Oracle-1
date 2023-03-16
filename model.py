from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization


class MyModel:
    def __init__(self, input_shape, num_classes):
        self.model = Sequential([
        # First block
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(strides=(2, 2)),
        Dropout(0.2),

        # Second block
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(strides=(2, 2)),
        Dropout(0.2),

        # Third block
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(strides=(2, 2)),
        Dropout(0.2),

        # Fourth block 
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(strides=(2, 2)),
        Dropout(0.2),

        # Fifth block
        Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(strides=(2, 2)),
        Dropout(0.2),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
])

    def get_model(self):
        return self.model
    
    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def predict(self, test_generator):
        probabilities = self.model.predict(test_generator)
        return probabilities
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def summary(self):
        self.model.summary()
    
    def get_history(self):
        return self.history.history