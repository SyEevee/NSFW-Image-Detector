# NSFW image detector using CNN
# Using InceptionV3 as base layer

import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import  Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint

# Using GPU
with tf.device('/gpu:0'):
    # Image dimensions
    img_width, img_height = 400, 400
    

    # Dataset paths
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    test_data_dir = 'data/test'

    # Parameters
    n_train_samples = 27000


    # Epochs
    epochs = 10
    batch_size = 5

    # Input shape
    input_shape = (img_width, img_height, 3)

    # Model architecture

    # base layer of inception model
    conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    # make the layers trainable
    for layer in conv_base.layers:
        layer.trainable = False
        
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    Dropout(0.3),
    model.add(Dense(5, activation='softmax'))

    # Model summary
    model.summary()

    # Model compilation

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range = 0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

    # Validation generator
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)
    

    # Model checkpoint
    filepath = "nsfw_model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # Model training
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        batch_size=batch_size,
        callbacks=[checkpoint]
        )

    # Model evaluation
    scores = model.evaluate_generator(test_generator)
    print("Accuracy = ", scores[1])

