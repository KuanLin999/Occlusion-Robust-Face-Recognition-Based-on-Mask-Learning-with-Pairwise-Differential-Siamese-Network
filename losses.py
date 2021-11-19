import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy

def cls_loss(y_true,y_pred):
    return tf.reduce_mean(categorical_crossentropy(y_true, y_pred))

def different_loss(Nonocc_feature, Occ_feature):
    return tf.reduce_mean(tf.abs(Nonocc_feature - Occ_feature))

