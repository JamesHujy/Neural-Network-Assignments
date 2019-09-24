from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d

import pickle
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss',default='softmax',type=str,required=False, description='select the loss function from softmax or euclidean')
    parser.add_argument('--layer', default=2,type=int, required=False)
    parser.add_argument('--hidden_size', default=256, type=int, required=False)
    parser.add_argument('--activation', default='relu', type=str, required=False)
    parser.add_argument('--learning_rate', default=1e-4, type=int, required=False)
    parser.add_argument('--weight_decay', default=1e-3, type=int, required=False)
    parser.add_argument('momentum', default=0.9, type=float, required=False)
    parser.add_argument('batch_size', default=100, type=int, required=False)
    parser.add_argument('max_epoch', default=100, type=int, required=False)
    parser.add_argument('disp_freq', default=100, type=int, required=False)
    parser.add_argument('test_epoch', default=5, type=int, required=False)
    args = parser.parse_args()
    return parser

def get_model(args):
    loss_dict = {}
    SoftmaxLoss = SoftmaxCrossEntropyLoss("softmax")
    EuclideanLoss = EuclideanLoss("euclidean")
    loss_dict['softmax'] = SigmoidLoss
    loss_dcit['euclidean'] = EuclideanLoss

    activation_dict = {}
    Relu = Relu("relu")
    Sigmoid = Sigmoid("sigmoid")
    activation_dict['relu'] = Relu
    activation_dict['sigmoid'] = Sigmoid

    
    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'batch_size': args.batch_size,
        'max_epoch': args.max_epoch,
        'disp_freq': args.disp_freq,
        'test_epoch': args.test_epoch
    }
    loss = loss_dict[args.loss]

    model = Network()
    if layer==1:
        model.add(Linear('fc1', 784, 10, 0.001))
        model.add(activation_dict[args.activation])
        
    elif layer == 2:
        model.add(Linear('fc1', 784, args.hidden_size, 0.001))
        model.add(activation_dict[args.activation])
        model.add(Linear('fc2', args.hidden_size, 10, 0.001))
        model.add(activation_dict[args.activation])
    else:
        model.add(Linear('fc1', 784, args.hidden_size, 0.001))
        model.add(activation_dict[args.activation])
        model.add(Linear('fc2', args.hidden_size, args.hidden_size//2, 0.001))
        model.add(activation_dict[args.activation])
        model.add(Linear('fc2', args.hidden_size//2, 10, 0.001))
        model.add(activation_dict[args.activation])
    return model, config, loss

def main():
    loss_list = []
    acc_list = []

    args = get_parser()

    model, config, loss = get_model(args)
    for epoch in range(1):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_loss, test_acc = test_net(model, loss, test_data, test_label, config['batch_size'])
        loss_list.append(test_loss)
        acc_list.append(test_acc)

def save(name, loss_list, acc_list):
    np.save('loss_'+name, loss_list)
    np.save('acc_'+name, acc_list)

if __name__ == '__main__':
    main()


'''
def one_layer_with_Sigmoid():
    model = Network()
    model.add(Linear('fc1', 784, 10, 0.001))
    model.add(Sigmoid('Sigmoid'))
    config = {
        'learning_rate': 0.0001,
        'weight_decay': 0.001,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 100,
        'test_epoch': 5
    }
    loss = EuclideanLoss(name='loss')
    return model, config, loss

def one_layer_with_Relu():
    model = Network()
    model.add(Linear('fc1', 784, 10, 0.001))
    model.add(Relu('Relu1'))
    config = {
        'learning_rate': 0.00001,
        'weight_decay': 0.001,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 100,
        'test_epoch': 5
    }
    loss = EuclideanLoss(name='loss')
    return model, config, loss

def two_layer_with_Sigmoid_with_Euclid(hidden_size):
    model = Network()
    model.add(Linear('fc1', 784, hidden_size, 0.001))
    model.add(Sigmoid("Sigmoid"))
    model.add(Linear('hidden', hidden_size, 10, 0.001))
    model.add(Sigmoid("Sigmoid"))
    config = {
        'learning_rate': 0.001,
        'weight_decay': 0.005,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 100,
        'test_epoch': 5
    }
    loss = EuclideanLoss(name='loss')
    return model, config, loss

def two_layer_with_Relu_with_Euclid(hidden_size):
    model = Network()
    model.add(Linear('fc1', 784, hidden_size, 0.001))
    model.add(Relu("Sigmoid"))
    model.add(Linear('hidden', hidden_size, 10, 0.001))
    model.add(Relu("Sigmoid"))
    config = {
        'learning_rate': 0.0001,
        'weight_decay': 0.005,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 100,
        'test_epoch': 5
    }
    loss = EuclideanLoss(name='loss')
    return model, config, loss

def two_layer_with_Relu_with_Softmax(hidden_size):
    model = Network()
    model.add(Linear('fc1', 784, hidden_size, 0.001))
    model.add(Relu("Sigmoid"))
    model.add(Linear('hidden', hidden_size, 10, 0.001))
    model.add(Relu("Sigmoid"))
    config = {
        'learning_rate': 0.0001,
        'weight_decay': 0.005,
        'momentum': 0.9,
        'batch_size': 100,
        'max_epoch': 100,
        'disp_freq': 100,
        'test_epoch': 5
    }
    loss = SoftmaxCrossEntropyLoss(name='loss')
    return model, config, loss
# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

'''

