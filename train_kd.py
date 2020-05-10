from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, datasets, models

from custom_functions import custom_cross_entrophy, softmax_with_temp, kl_divergence_cross_entrophy
from models import vgg_2blocks

AUTO = tf.data.experimental.AUTOTUNE

TEMP = 5
BATCH_SIZE = 128
EPOCHS = 100
TRAIN_LOG_DIR = Path('logs/KD_DL_3/train')
TEST_LOG_DIR = Path('logs/KD_DL_3/test')
SAVE_PATH = Path('checkpoints/KD_DL_3/')
CHECKPOINT_PATH = Path('checkpoints/KD_DL_2/1')
LOAD_FROM_CHECKPOINTS = True


def main():

    (train_images, train_labels), (test_images,
                                   test_labels) = datasets.cifar10.load_data()
    teacher_soft_logits = np.load("teacher_softlogits.npy")
    print('loading done')
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    train_labels = tf.keras.utils.to_categorical(train_labels.astype('float32'))
    test_labels = tf.keras.utils.to_categorical(test_labels.astype('float32'))
  
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels, teacher_soft_logits))
    dataset = dataset.repeat(EPOCHS).batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)

    # make student model
    
    if LOAD_FROM_CHECKPOINTS:
        student_model = tf.saved_model.load(str(CHECKPOINT_PATH))
        print(list(student_model.signatures.keys()))
    else:
        student_model = vgg_2blocks()
    # student_model = tf.keras.Model(
    #     inputs=models.input, outputs=models.get_layer('logits').output)

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

    # Define our metrics
    train_acc = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
    test_acc = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

    train_summary_writer = tf.summary.create_file_writer(str(TRAIN_LOG_DIR))
    test_summary_writer = tf.summary.create_file_writer(str(TEST_LOG_DIR))
    cross_entropy = tf.keras.losses.CategoricalCrossentropy()
    soft_kl_divergence = tf.keras.losses.KLDivergence()
    def train_step(images, labels, teacher_soft_logits):

        with tf.GradientTape() as tape:
            pred = student_model(images, training=True)

            unsoft_pred = softmax_with_temp(pred, 1)
            soft_pred = softmax_with_temp(pred, TEMP)

            teacher_logits = teacher_soft_logits
            softened_teacher_prob = softmax_with_temp(teacher_logits, TEMP)
            
            # loss_value = custom_cross_entrophy(
            #     labels, softened_teacher_prob, unsoft_pred, soft_pred)
            
            loss_value = kl_divergence_cross_entrophy(labels, softened_teacher_prob, unsoft_pred, soft_pred, cross_entropy,soft_kl_divergence,alpha=0.4, temp=TEMP)

        grads = tape.gradient(loss_value, student_model.trainable_variables)
        opt.apply_gradients(zip(grads, student_model.trainable_variables))

        train_acc(labels, pred)
        train_loss(loss_value)

        return loss_value

    @tf.function
    def train():

        step = 0
        ckpt_step = 0
        ckpt_step = tf.cast(ckpt_step, tf.int64)

        for x, y, soft_logits in dataset:

            _ = train_step(x, y, soft_logits)
            step += 1
            # End of one epoch
            if step % int(len(train_images) / BATCH_SIZE) == 0:

                test_acc(test_labels, student_model(
                    test_images, training=False))
                tf.print("Steps loss     accuracy   test_accuracy")
                tf.print(step, train_loss.result(),
                         train_acc.result(), test_acc.result())

                ckpt_step += 1
                with train_summary_writer.as_default():
                    tf.summary.scalar('epoch_accuracy',
                                      train_acc.result(), step=ckpt_step)
                    tf.summary.scalar(
                        'epoch_loss', train_loss.result(), step=ckpt_step)
                with test_summary_writer.as_default():
                    tf.summary.scalar('epoch_accuracy',
                                      test_acc.result(), step=ckpt_step)

                train_acc.reset_states()
                test_acc.reset_states()
                train_loss.reset_states()

    train()
    tf.saved_model.save(student_model, str(SAVE_PATH))


if __name__ == "__main__":
    main()
