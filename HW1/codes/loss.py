from __future__ import division
import numpy as np
from load_data import load_mnist_2d
import time
class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        distance = (target - input)** 2
        loss = distance.mean(axis=0).sum()/2
        return loss

    def backward(self, input, target):
        return (input - target) / len(input)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        input_exp = np.exp(input)
        input_sum = np.sum(input_exp, axis=1).reshape(-1,1)     
        softmax = input_exp / input_sum
        log_softmax = np.log(softmax)

        loss = ((-log_softmax*target).sum(axis=-1)).mean(axis=-1)      
        return loss

    def backward(self, input, target):
        input_exp = np.exp(input)
        input_sum = np.sum(input_exp, axis=1).reshape(-1,1)
        softmax = input_exp / input_sum
        return (softmax - target) / len(input)

