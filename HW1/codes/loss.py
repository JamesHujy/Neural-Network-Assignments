from __future__ import division
import numpy as np
from load_data import load_mnist_2d

class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        distance = (target - input)** 2
        loss = distance.mean(axis=0).sum()/2
        return loss

    def backward(self, input, target):
        return target - input

class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        input_exp = np.exp(input)
        input_sum = np.sum(input_exp, axis=1).reshape(len(input),1)
        
        softmax = input_exp / input_sum
        loss = softmax.dot(target.T).sum(axis=0)/len(input)
        return loss

    def backward(self, input, target):
        input_exp = np.exp(input)
        input_sum = np.sum(input_exp, axis=-1).reshape(len(input),1)
        softmax = input_exp / input_sum
        return softmax - target
