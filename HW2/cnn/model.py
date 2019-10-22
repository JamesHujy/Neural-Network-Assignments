# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class Model:
    def __init__(self,
                 learning_rate=0.001,
                 dropout=0.5,
                 batch_norm=True,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.y_ = tf.placeholder(tf.int32, [None])

        # TODO:  fill the blank of the arguments
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.loss, self.pred, self.acc = self.forward(True)
        self.loss_val, self.pred_val, self.acc_val = self.forward(False, reuse=True)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        
        # TODO:  maybe you need to update the parameter of batch_normalization?
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=None):
    
        with tf.variable_scope("model", reuse=reuse):
            X = tf.layers.conv2d(self.x_, filters=256, kernel_size=3, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            if self.batch_norm:
                X = batch_normalization_layer(X, is_train=is_train)
            X = tf.nn.relu(X)
            X = dropout_layer(X, self.dropout, is_train=is_train)
            X = tf.layers.max_pooling2d(X, pool_size=2, strides=2)
            
            X = tf.layers.conv2d(X, filters=128, kernel_size=4, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            if self.batch_norm:
                X = batch_normalization_layer(X, is_train=is_train)
            X = tf.nn.relu(X)
            X = dropout_layer(X, self.dropout, is_train=is_train)
            X = tf.layers.max_pooling2d(X, pool_size=2, strides=2)
            
            output = tf.reshape(X, [-1, 6*6*128])
            logits = tf.layers.dense(output, 10, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        
        return loss, pred, acc

def batch_normalization_layer(incoming, is_train=True):
    return tf.layers.batch_normalization(incoming, 3, training=is_train)
    
def dropout_layer(incoming, drop_rate, is_train=True):
    return tf.layers.dropout(incoming, drop_rate, training=is_train)
