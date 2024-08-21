import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(train_dir, validation_dir):
    """Load and preprocess training and validation data."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),  # Ensure all images are resized to 150x150
        batch_size=20,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),  # Ensure all images are resized to 150x150
        batch_size=20,
        class_mode='binary'
    )

    return train_generator, validation_generator