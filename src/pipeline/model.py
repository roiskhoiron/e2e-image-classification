from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, Sequential


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (150, 150, 3),
    num_classes: int = 3,
    dropout_rate: float = 0.5,
    dense_units: int = 128,
) -> Sequential:
    model = Sequential(name="image_classifier_cnn")
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Block 4
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    # Classification head
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


def compile_model(
    model: Sequential,
    learning_rate: float = 0.001,
) -> Sequential:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(
    checkpoint_path: str = "models_output/checkpoints/best.keras",
    patience: int = 5,
    monitor: str = "val_loss",
) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_model(
    model: Sequential,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 50,
    checkpoint_path: str = "models_output/checkpoints/best.keras",
) -> tf.keras.callbacks.History:
    callbacks = get_callbacks(checkpoint_path=checkpoint_path)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history
