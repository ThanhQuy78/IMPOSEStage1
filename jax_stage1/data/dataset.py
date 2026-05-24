"""Dataset loader for NIST SD302a rolled fingerprints.

Loads grayscale PNG fingerprint images from the challengers directory,
preprocesses them (resize to 512x512, normalize to [-1, 1]),
and returns batches suitable for JAX/TPU training.

Uses tf.data for efficient I/O with prefetching and shuffling.
"""

import os
import glob
import math
import numpy as np
import jax
import jax.numpy as jnp
from typing import Iterator, Tuple, Optional


def find_all_images(dataset_path: str) -> list:
    """Recursively find all PNG images in the challengers directory.
    
    Expected structure: dataset_path/{A-H}/roll/png/*.png
    
    Args:
        dataset_path: Path to the challengers directory.
        
    Returns:
        Sorted list of absolute image paths.
    """
    patterns = [
        os.path.join(dataset_path, "**", "*.png"),
    ]
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern, recursive=True))
    
    # Filter to only include 'roll' images (skip flat/slap if present)
    roll_files = [f for f in all_files if os.sep + "roll" + os.sep in f 
                  or "/roll/" in f]
    
    # If no 'roll' filter matches, use all files
    if not roll_files:
        roll_files = all_files
    
    roll_files.sort()
    return roll_files


def load_and_preprocess_image(image_path: str, target_size: int = 512) -> np.ndarray:
    """Load a single image and preprocess it.
    
    Args:
        image_path: Path to PNG image.
        target_size: Target spatial size (default 512).
        
    Returns:
        Preprocessed image as float32 array, shape (H, W, 1), range [-1, 1].
    """
    try:
        from PIL import Image
    except ImportError:
        import cv2
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 127.5 - 1.0
        return img[:, :, np.newaxis]
    
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize((target_size, target_size), Image.BILINEAR)
    img = np.array(img, dtype=np.float32)
    img = img / 127.5 - 1.0  # Normalize to [-1, 1]
    return img[:, :, np.newaxis]  # (H, W, 1)


def create_numpy_dataset(
    dataset_path: str,
    image_size: int = 512,
    batch_size: int = 48,
    shuffle: bool = True,
    seed: int = 42,
    cache_images: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load entire dataset into memory as numpy array.
    
    For 13,630 images at 512x512x1 float32 ≈ 13.4 GB.
    If memory is tight, use create_streaming_dataset instead.
    
    Args:
        dataset_path: Path to challengers directory.
        image_size: Target image size.
        batch_size: Batch size (should be divisible by num_devices).
        shuffle: Whether to shuffle.
        seed: Random seed.
        cache_images: Whether to cache all images in memory.
        
    Returns:
        Tuple of (images array of shape (N, H, W, 1), num_images).
    """
    image_paths = find_all_images(dataset_path)
    num_images = len(image_paths)
    print(f"Found {num_images} images in {dataset_path}")
    
    if num_images == 0:
        raise ValueError(f"No images found in {dataset_path}")
    
    if cache_images:
        print(f"Loading all {num_images} images into memory...")
        images = np.zeros((num_images, image_size, image_size, 1), dtype=np.float32)
        for i, path in enumerate(image_paths):
            if i % 1000 == 0:
                print(f"  Loading {i}/{num_images}...")
            images[i] = load_and_preprocess_image(path, image_size)
        print(f"Dataset loaded: {images.shape}, {images.nbytes / 1e9:.1f} GB")
        return images, num_images
    
    return image_paths, num_images


def data_iterator(
    images: np.ndarray,
    batch_size: int,
    num_devices: int = 8,
    shuffle: bool = True,
    rng: Optional[np.random.RandomState] = None,
) -> Iterator[jnp.ndarray]:
    """Create a data iterator that yields device-sharded batches.
    
    Each batch has shape (num_devices, per_device_batch, H, W, C)
    for use with jax.pmap.
    
    Args:
        images: Full dataset array (N, H, W, 1).
        batch_size: Total batch size across all devices.
        num_devices: Number of TPU/GPU devices.
        shuffle: Whether to shuffle each epoch.
        rng: Random state for shuffling.
        
    Yields:
        Batches of shape (num_devices, batch_size//num_devices, H, W, C).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    num_images = len(images)
    per_device_batch = batch_size // num_devices
    assert batch_size % num_devices == 0, \
        f"batch_size ({batch_size}) must be divisible by num_devices ({num_devices})"
    
    # Drop last incomplete batch
    steps_per_epoch = num_images // batch_size
    
    while True:
        if shuffle:
            perm = rng.permutation(num_images)
        else:
            perm = np.arange(num_images)
        
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size: (step + 1) * batch_size]
            batch = images[idx]  # (batch_size, H, W, 1)
            
            # Reshape for pmap: (num_devices, per_device_batch, H, W, C)
            batch = batch.reshape(num_devices, per_device_batch, *batch.shape[1:])
            
            yield jnp.array(batch)


def create_tf_dataset(
    dataset_path: str,
    image_size: int = 512,
    batch_size: int = 48,
    num_devices: int = 8,
    shuffle_buffer: int = 4096,
    seed: int = 42,
):
    """Create a tf.data pipeline for efficient data loading.
    
    This is more memory-efficient than loading everything into RAM.
    
    Args:
        dataset_path: Path to challengers directory.
        image_size: Target image size.
        batch_size: Total batch size across all devices.
        num_devices: Number of devices for sharding.
        shuffle_buffer: Size of shuffle buffer.
        seed: Random seed.
        
    Returns:
        tf.data.Dataset yielding batches of shape 
        (num_devices, per_device_batch, H, W, 1).
    """
    import tensorflow as tf
    
    image_paths = find_all_images(dataset_path)
    num_images = len(image_paths)
    print(f"Found {num_images} images for tf.data pipeline")
    
    per_device_batch = batch_size // num_devices
    
    def load_image(path):
        raw = tf.io.read_file(path)
        img = tf.image.decode_png(raw, channels=1)  # (H, W, 1)
        img = tf.image.resize(img, [image_size, image_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        return img
    
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()
    
    def reshape_for_pmap(batch):
        return tf.reshape(batch, [num_devices, per_device_batch, image_size, image_size, 1])
    
    ds = ds.map(reshape_for_pmap)
    
    return ds, num_images


def get_steps_per_epoch(num_images: int, batch_size: int) -> int:
    """Compute number of training steps per epoch."""
    return num_images // batch_size
