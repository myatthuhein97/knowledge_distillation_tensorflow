from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, models

(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

print(train_images.shape)

MODEL_PATH = Path("checkpoints/cifar10_vgg4")

# You can turn this on if you don't have enough memory

# batch_size = 256
# dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
# dataset = dataset.batch(batch_size)
# dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

teacher_model = models.load_model(str(MODEL_PATH))



teacher_model.summary()

results = teacher_model.predict(train_images)
np.save("teacher_softlogits.npy", results)
print(results.shape)
