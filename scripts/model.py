#Define the model
model = tf.keras.Sequential()

# Adding convolutional layers. To extract features from the input images
# The Conv2D layers are responsible for learning features from the input images using convolution operations. They use a rectified linear unit (ReLU) activation function to introduce non-linearity.
# The MaxPooling2D layers Calculates the maximum value of patches of a feature map, creating a downsample feature map.
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Dropout(0.25)) # Introducing dropout regularizer to mitigate overfitting
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#Add flattening Layer; reshapes the output from the previous layer into a 1D vector.
#This is necessary to connect the convolutional layers to fully connected layers.
model.add(layers.Flatten())

#Add Dense(Fully Connected Layers) to make predictions
model.add(layers.Dense(64, activation='relu')) #consists of 64 units (neurons) with ReLU activation. It helps the model learn more complex representations of the data.
model.add(layers.Dense(10))  # 10 output classes consists of 10 units, corresponding to the number of output classes in the CIFAR-10 dataset. It has no activation function specified, as it will be used for the final output logits.

model.summary() #prints a summary of the model architecture, including the type of each layer, output shape, and the number of parameters.
