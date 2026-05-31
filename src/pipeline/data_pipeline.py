import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def build_dataset(
    directory: str,
    image_size: tuple = (150, 150),
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = False,
    buffer_size: int = 1000,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from a directory of images.

    Parameters
    ----------
    directory : str
        Path to dataset folder with subfolders per class.
    image_size : tuple
        Target (height, width) for resizing.
    batch_size : int
        Number of samples per batch.
    augment : bool
        Apply data augmentation (training only).
    shuffle : bool
        Shuffle the dataset.
    buffer_size : int
        Buffer size for shuffle and prefetch.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tf.data.Dataset
        Batched and optimized dataset pipeline.
    """
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    # Unbatch so we can apply per-sample operations
    ds = ds.unbatch()
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed)
    # Map: cast image to float32 and rescale
    def _rescale(image, label):
        return tf.cast(image, tf.float32) / 255.0, label
    ds = ds.map(_rescale, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if augment:
        ds = ds.map(_apply_augmentation, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _apply_augmentation(image, label):
    # Random horizontal & vertical flip
    image = tf.image.random_flip_left_right(image, seed=42)
    image = tf.image.random_flip_up_down(image, seed=43)
    # Random rotation by multiples of 90 degrees
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    # Random brightness & contrast
    image = tf.image.stateless_random_brightness(image, max_delta=0.1, seed=(44, 45))
    image = tf.image.stateless_random_contrast(image, lower=0.8, upper=1.2, seed=(46, 47))
    # Random zoom (scale up/down) then crop/pad back to original size
    zoom = tf.random.uniform([], 0.9, 1.1)
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    h_new = tf.cast(h / zoom, tf.int32)
    w_new = tf.cast(w / zoom, tf.int32)
    image = tf.image.resize(image, [h_new, w_new])
    image = tf.image.resize_with_crop_or_pad(image, tf.cast(h, tf.int32), tf.cast(w, tf.int32))
    return image, label


def get_train_val_test(
    data_dir: str = "data/processed",
    image_size: tuple = (150, 150),
    batch_size: int = 32,
) -> tuple:
    """Convenience function to build all three splits."""
    train_ds = build_dataset(
        f"{data_dir}/train",
        image_size=image_size,
        batch_size=batch_size,
        augment=True,
        shuffle=True,
    )
    val_ds = build_dataset(
        f"{data_dir}/val",
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
    )
    test_ds = build_dataset(
        f"{data_dir}/test",
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
    )
    return train_ds, val_ds, test_ds
