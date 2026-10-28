import tensorflow as tf
import tensorflow_datasets as tfds
from utils import is_cat_breed

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    return image, label

def load_dataset():
    dataset, info = tfds.load("oxford_iiit_pet", with_info=True, as_supervised=True)
    train_data = dataset['train'].map(preprocess).shuffle(1000).batch(BATCH_SIZE)
    test_data = dataset['test'].map(preprocess).batch(BATCH_SIZE)
    return train_data, test_data, info

def load_cat_dataset(val_every=7):
    """Oxford-IIIT Pet filtered to the 12 cat breeds, labels remapped to 0..11.

    Images are 224x224 float32 in 0..255 (the v2 model does its own preprocessing).
    Every `val_every`-th training image is held out for validation.
    Returns (train, val, test, cat_class_names, n_train_images).
    """
    dataset, info = tfds.load("oxford_iiit_pet", with_info=True, as_supervised=True)
    all_names = info.features['label'].names
    cat_ids = [i for i, name in enumerate(all_names) if is_cat_breed(name)]
    cat_names = [all_names[i] for i in cat_ids]
    id_map = tf.constant(
        [cat_ids.index(i) if i in cat_ids else -1 for i in range(len(all_names))],
        dtype=tf.int64,
    )

    def is_cat(image, label):
        return tf.gather(id_map, label) >= 0

    def to_small(image, label):
        # uint8 keeps the in-memory cache 4x smaller
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        return tf.cast(tf.round(image), tf.uint8), tf.cast(tf.gather(id_map, label), tf.int32)

    def to_float(image, label):
        return tf.cast(image, tf.float32), label

    def prepare(ds):
        return ds.filter(is_cat).map(to_small, AUTOTUNE).cache()

    indexed = dataset['train'].enumerate()
    train = prepare(indexed.filter(lambda i, ex: i % val_every != 0).map(lambda i, ex: ex))
    val = prepare(indexed.filter(lambda i, ex: i % val_every == 0).map(lambda i, ex: ex))
    test = prepare(dataset['test'])

    # One counting pass also fills the cache, so epochs afterwards are fast
    n_train = int(train.reduce(0, lambda count, _: count + 1))

    train = train.shuffle(1000, seed=42).map(to_float, AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val = val.map(to_float, AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test = test.map(to_float, AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return train, val, test, cat_names, n_train
