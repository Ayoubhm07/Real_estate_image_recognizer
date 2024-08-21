from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from process import load_data


def build_model():
    """Constructs and returns the image classification model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


def train_model(train_dir, validation_dir):
    """Loads data, builds, compiles, and trains the model."""
    train_generator, validation_generator = load_data(train_dir, validation_dir)
    model = build_model()

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping to avoid overfitting
    early_stopping_monitor = EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=5,
        verbose=1,
        restore_best_weights=True
    )

    # Fit the model
    model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        verbose=2,
        callbacks=[early_stopping_monitor]
    )

    # Save the trained model
    model.save('my_model.keras')


if __name__ == "__main__":
    train_dir = '../data/train'
    validation_dir = '../data/validation'
    train_model(train_dir, validation_dir)
