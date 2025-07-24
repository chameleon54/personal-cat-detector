import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    return image, label

def load_dataset():
    dataset, info = tfds.load("oxford_iiit_pet", with_info=True, as_supervised=True)
    train_data = dataset['train'].map(preprocess).shuffle(1000).batch(BATCH_SIZE)
    test_data = dataset['test'].map(preprocess).batch(BATCH_SIZE)
    return train_data, test_data, info
