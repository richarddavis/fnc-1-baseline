# Not working...

from keras.metrics import binary_accuracy, categorical_accuracy
import tensorflow as tf

def fnc_score(y_true, y_pred):
    "Assumes two outputs: related and stance"
    y_true_related, y_true_stance = tf.split(y_true, 2, axis=1)
    y_pred_related, y_pred_stance = tf.split(y_pred, 2, axis=1)
    print(y_true_related, y_true_stance, y_pred_related, y_pred_stance)

    a1 = categorical_accuracy(y_true_related, y_pred_related)
    a2 = categorical_accuracy(y_true_stance, y_pred_stance)

    print(a1, a2)
    return (
        tf.multiply(a1, 0.25) + 
        tf.multiply(a2, 0.75)
        #tf.multiply(categorical_accuracy(y_true_related, y_pred_related), 0.25) + 
        #tf.multiply(categorical_accuracy(y_true_stance, y_pred_stance), 0.75)
    )
