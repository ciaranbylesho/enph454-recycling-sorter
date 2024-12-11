import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def create_cnn(num_conv_layers, num_filters, kernel_size, num_fc_units, input_shape):
    """
    Creates a simple CNN model based on the specified parameters.

    Parameters:
    - num_conv_layers (int): The number of convolutional layers.
    - num_filters (int): The number of filters in each convolutional layer.
    - kernel_size (tuple): The kernel size for convolutional layers, e.g. (3, 3).
    - num_fc_units (int): The number of units in the fully connected layer.
    - input_shape (tuple): Shape of the input data, e.g., (224, 224, 3).

    Returns:
    - model (tensorflow.keras.Model): The compiled CNN model.
    """
    model = models.Sequential()

    # Add convolutional layers
    for i in range(num_conv_layers):
        if i == 0:
            model.add(layers.Conv2D(num_filters, kernel_size, activation='relu', input_shape=input_shape))
        else:
            model.add(layers.Conv2D(num_filters, kernel_size, activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))  # Max pooling after each conv layer

    # Flatten the output of the convolutional layers before passing it to the fully connected layer
    model.add(layers.Flatten())

    # Add a fully connected (dense) layer
    model.add(layers.Dense(num_fc_units, activation='relu'))

    # Add the output layer (assume binary classification, adjust for your use case)
    model.add(layers.Dense(1, activation='sigmoid'))  # Use 'softmax' for multi-class problem

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_cnn_multiclass(num_conv_layers, num_filters, kernel_size, num_fc_units, input_shape, output_size):
    """
    Creates a simple CNN model based on the specified parameters.

    Parameters:
    - num_conv_layers (int): The number of convolutional layers.
    - num_filters (int): The number of filters in each convolutional layer.
    - kernel_size (tuple): The kernel size for convolutional layers, e.g. (3, 3).
    - num_fc_units (int): The number of units in the fully connected layer.
    - input_shape (tuple): Shape of the input data, e.g., (224, 224, 3).

    Returns:
    - model (tensorflow.keras.Model): The compiled CNN model.
    """
    model = models.Sequential()

    # Add convolutional layers
    for i in range(num_conv_layers):
        if i == 0:
            model.add(layers.Conv2D(num_filters, kernel_size, activation='relu', input_shape=input_shape))
        else:
            model.add(layers.Conv2D(num_filters, kernel_size, activation='relu'))
            if i % 2 == 0:
                model.add(layers.Dropout(0.5))
        model.add(layers.MaxPooling2D((2, 2)))  # Max pooling after each conv layer

    # Flatten the output of the convolutional layers before passing it to the fully connected layer
    model.add(layers.Flatten())

    # Add a fully connected (dense) layer
    model.add(layers.Dense(num_fc_units, activation='relu'))

    # Add the output layer (assume binary classification, adjust for your use case)
    model.add(layers.Dense(output_size, activation='softmax'))  # Use 'softmax' for multi-class problem

    # Compile the model
    optimizer = Adam(learning_rate=0.0005, clipvalue=1.0)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model