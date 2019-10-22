import numpy as np
import matplotlib.pyplot as plt

train_acc = np.load('./train_info/train_acc.npy')
print(train_acc.shape)

val_acc = np.load('./train_info/val_acc.npy')
print(val_acc.shape)


def drawloss():
	plt.plot(train_acc)

    