import numpy as np
import tensorflow as tf
from common import config

mean, std = config.mean, config.std

class Attack():

    def __init__(self, model, batchsize, alpha, beta, gamma, CW_kappa, use_cross_entropy_loss, **kwargs):
        self.batchsize = batchsize
        self.model = model  # pretrained vgg model used as classifier
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.CW_kappa = CW_kappa
        self.use_cross_entropy_loss = use_cross_entropy_loss
    
    '''Build computation graph for generating adversarial examples'''
    def generate_graph(self, pre_noise, x, gt, target=None, **kwargs):
        noise = 10 * tf.tanh(pre_noise) 
        x_noise = x + noise                 ## add perturbation and get adversarial examples
        x_clip = tf.clip_by_value(x_noise, 0, 255) 
        x_round = x_clip + tf.stop_gradient(x_clip // 1 - x_clip) ##skip computing gradient wrt to rounded results(x_round) and only calculate the gradient wrt to x_clip 
        x_norm = (x_round - mean)/(std + 1e-7)          ## normalize the image input for the classfier
        logits = self.model.build(x_norm)               

        if self.use_cross_entropy_loss:
            loss = self.cross_entropy_loss(gt, target, logits, target is not None)
        else:
            loss = self.CW_attack_loss(gt, target, logits, target is not None)
        reg = self.regularization((x_round - x) / 255)
        loss = tf.reduce_mean(self.alpha * loss + reg)

        preds = tf.nn.softmax(logits)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32), gt), tf.float32))
        return acc, loss, x_round

    def regularization(self, pertubations):
        L1_reg = tf.contrib.layers.l1_regularizer(1.0)(pertubations)
        L2_reg = tf.contrib.layers.l2_regularizer(1.0)(pertubations)
        return self.beta*L1_reg + self.gamma*L2_reg

    def cross_entropy_loss(self, ground_truth, target, logits, is_targeted_attack):
        if is_targeted_attack:
            target_labels = tf.one_hot(target, config.nr_class, dtype=tf.int32)
            softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=target_labels)
            loss = tf.reduce_sum(softmax, axis=0)
        else: 
            gt_labels = tf.one_hot(ground_truth, config.nr_class, dtype=tf.int32)
            softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=gt_labels)
            loss = -tf.reduce_sum(softmax, axis=0)
        return loss

    def CW_attack_loss(self, ground_truth, target, logits, is_targeted_attack):
        if is_targeted_attack:
            target_labels = tf.one_hot(target, config.nr_class, dtype=tf.int32)
            mask = tf.cast(target_labels, tf.float32)
            target_logits = tf.reduce_max(mask * logits - (1 - mask) * 1e8, axis=1)
            non_target_logits = tf.reduce_max((1 - mask) * logits - mask * 1e8, axis=1)
            loss = -tf.reduce_sum(tf.minimum(target_logits - non_target_logits, self.CW_kappa))
        else: 
            gt_labels = tf.one_hot(ground_truth, config.nr_class, dtype=tf.int32)
            mask = tf.cast(gt_labels, tf.float32)
            true_logit = tf.reduce_max(mask*logits - (1 - mask) * 1e8, axis=1)
            false_logit = tf.reduce_max((1 - mask) * logits - mask * 1e8, axis=1)
            loss = tf.reduce_sum(tf.minimum(true_logit - false_logit, -self.CW_kappa))            
        return loss

    '''Build a graph for evaluating the classification result of adversarial examples'''
    def evaluate(self, x, gt, **kwargs): 
        x = (x - mean)/(std + 1e-7)
        logits = self.model.build(x)
        preds = tf.nn.softmax(logits)
        gt_one_hot = tf.one_hot(gt, config.nr_class)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                             tf.cast(tf.argmax(gt_one_hot, 1), dtype=tf.int32)), tf.float32))
        return acc
