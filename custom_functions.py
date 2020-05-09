import numpy as np
import tensorflow as tf


def softmax_with_temp(logits, temp=1):

    logits = (logits - tf.math.reduce_max(logits)) / temp
    exp_logits = tf.math.exp(logits)
    logits_sum = tf.math.reduce_sum(exp_logits, axis=-1, keepdims=True)
    result = exp_logits / logits_sum

    return result


def custom_cross_entrophy(y_true, y_soft, y_pred, y_soft_pred, alpha=0.5):

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    cross_entropy = -tf.math.reduce_mean(tf.math.reduce_sum(
        y_true * tf.math.log(y_pred), axis=-1, keepdims=False))

    y_soft = tf.clip_by_value(y_soft, 1e-7, 1 - 1e-7)
    y_soft_pred = tf.clip_by_value(y_soft_pred, 1e-7, 1 - 1e-7)
    soft_cross_entropy = -tf.math.reduce_mean(tf.math.reduce_sum(
        y_soft * tf.math.log(y_soft_pred), axis=-1, keepdims=False))

    return alpha * soft_cross_entropy + (1 - alpha) * cross_entropy


def kl_divergence_cross_entrophy(y_true, y_soft, y_pred, y_soft_pred,cross_entropy,soft_kl_divergence, alpha=0.5,temp=1):

    cross_entropy = tf.keras.losses.CategoricalCrossentropy()

    soft_kl_divergence = tf.keras.losses.KLDivergence()

    return alpha * soft_kl_divergence(y_soft, y_soft_pred) + (1-alpha) * cross_entropy(y_true,y_pred)
