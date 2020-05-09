from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets

from models import vgg_3blocks, vgg_2blocks

EPOCH = 100
BATCH_SIZE = 256
VAL_FREQUENCY = 1
TENSORBOARD_LOG = Path("logs/cifar10_vgg2")
SAVE_MODEL_PATH = Path("checkpoints/cifar10_vgg2")

def lr_schedule(epoch):
    lr = 1e-3
    if epoch >= 85:
        lr *= 1e-1
    elif epoch >= 75:
        lr *= 1e-1
    elif epoch >= 55:
        lr *= 1e-1
    elif epoch >= 30:
        lr *= 1e-1
    return lr

def main():
    (train_images, train_labels), (test_images,
                                   test_labels) = datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1

    train_images, test_images = train_images / 255.0, test_images / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # You can define your own model in models.py
    model = vgg_2blocks()

    # You can define what you want to logs in this objects
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=TENSORBOARD_LOG)
    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=EPOCH, batch_size=BATCH_SIZE,
                        validation_data=(test_images, test_labels), callbacks=[tensorboard_callback], validation_freq=VAL_FREQUENCY)

    model.save(str(SAVE_MODEL_PATH))

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_acc)


if __name__ == "__main__":
    main()
