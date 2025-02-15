#Learning rate schedular, adjusts learning rate during training
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.9 ** epoch)

model.compile(optimizer='adam',  #Used for gradient-based optimization
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #Specifies loss function used to train classification models
              metrics=['accuracy'])  #During training, the model's accuracy on both the training and validation datasets will be calculated and displayed

history = model.fit(train_images, train_labels, epochs=70,
                    validation_data=(test_images, test_labels))
