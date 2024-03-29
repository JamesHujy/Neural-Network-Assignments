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
import datetime
import matplotlib.pyplot as plt

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss',default='softmax',type=str,required=False)
    parser.add_argument('--hidden_layer', default=1,type=int, required=False)
    parser.add_argument('--hidden_size', default=256, type=int, required=False)
    parser.add_argument('--activation', default='relu', type=str, required=False)
    parser.add_argument('--learning_rate', default=1e-4, type=float, required=False)
    parser.add_argument('--learning_rate_decay', default=1.,type=float,required=False)
    parser.add_argument('--weight_decay', default=1e-3, type=float, required=False)
    parser.add_argument('--momentum', default=0.9, type=float, required=False)
    parser.add_argument('--batch_size', default=100, type=int, required=False)
    parser.add_argument('--max_epoch', default=100, type=int, required=False)
    parser.add_argument('--disp_freq', default=100, type=int, required=False)
    parser.add_argument('--test_epoch', default=5, type=int, required=False)
    args = parser.parse_args()
    return args

def get_activation(activate_type, index):
    if activate_type == "relu":
        return Relu("Relu"+str(index))
    else:
        return Sigmoid("Sigmoid"+str(index))

def get_model(args):
    loss_dict = {}
    softmaxLoss = SoftmaxCrossEntropyLoss("softmax")
    euclideanLoss = EuclideanLoss("euclidean")
    loss_dict['softmax'] = softmaxLoss
    loss_dict['euclidean'] = euclideanLoss
    
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
    layer = args.hidden_layer
    if layer == 1:
        model.add(Linear('fc1', 784, args.hidden_size, 0.01))
        model.add(get_activation(args.activation, 0))
        model.add(Linear('fc2', args.hidden_size, 10, 0.01))
        model.add(get_activation(args.activation, 1))
    else:
        model.add(Linear('fc1', 784, args.hidden_size, 0.01))
        model.add(get_activation(args.activation, 0))
        model.add(Linear('fc2', args.hidden_size, args.hidden_size//2, 0.01))
        model.add(get_activation(args.activation, 1))
        model.add(Linear('fc2', args.hidden_size//2, 10, 0.01))
        model.add(get_activation(args.activation, 2))
    return model, config, loss

def save(args, train_loss_list, train_acc_list, test_loss_list, test_acc_list):
    prefix = './npy/'+'_'.join([args.loss, args.activation, "hidden-size", str(args.hidden_layer)])
    np.save(prefix + "_train_loss.npy", np.array(train_loss_list))
    np.save(prefix + "_train_acc.npy", np.array(train_acc_list))
    np.save(prefix + "_test_loss.npy", np.array(test_loss_list))
    np.save(prefix + "_test_acc.npy", np.array(test_acc_list))

def main():
    test_loss_list = []
    test_acc_list = []
    train_loss_list = []
    train_acc_list = []

    args = get_parser()

    model, config, loss = get_model(args)

    
    starttime = datetime.datetime.now()    
    for epoch in range(args.max_epoch):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        train_loss, train_acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        train_loss_list.extend(train_loss)
        train_acc_list.extend(train_acc)
        if epoch > 5:
            config['learning_rate'] = config['learning_rate'] * args.learning_rate_decay
        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_loss, test_acc = test_net(model, loss, test_data, test_label, config['batch_size'])
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
    endtime = datetime.datetime.now()
    print ("total training time:",(endtime - starttime).seconds)

    save(args, train_loss_list, train_acc_list, test_loss_list, test_acc_list)



if __name__ == '__main__':
    main()
