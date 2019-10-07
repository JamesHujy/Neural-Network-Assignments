import matplotlib.pyplot as plt
import numpy as np


acc = np.load('./npy/euclidean_relu_hidden-size_1_train_acc.npy')
acc = acc[range(0,len(acc),10)]
loss = np.load('./npy/euclidean_relu_hidden-size_1_train_loss.npy')
loss = loss[range(0,len(loss),10)]

def shorten(onelist, interval):
	returnlist = onelist[range(0,len(onelist), interval)]
	return returnlist

euclidean_relu_hidden_size_1_train_acc = np.load('./npy/euclidean_relu_hidden-size_1_train_acc.npy')

euclidean_relu_hidden_size_1_train_loss = np.load('./npy/euclidean_relu_hidden-size_1_train_loss.npy')

softmax_relu_hidden_size_1_train_acc = np.load('./npy/softmax_relu_hidden-size_1_train_acc.npy')
softmax_relu_hidden_size_1_train_loss = np.load('./npy/softmax_relu_hidden-size_1_train_loss.npy')

euclidean_sigmoid_hidden_size_1_train_acc = np.load('./npy/euclidean_sigmoid_hidden-size_1_train_acc.npy')
euclidean_sigmoid_hidden_size_1_train_loss = np.load('./npy/euclidean_sigmoid_hidden-size_1_train_loss.npy')

softmax_sigmiod_hidden_size_1_train_acc = np.load('./npy/softmax_relu_hidden-size_1_train_acc.npy')
softmax_sigmoid_hidden_size_1_train_loss = np.load('./npy/softmax_relu_hidden-size_1_train_loss.npy')


euclidean_sigmoid_hidden_size_1_test_acc = np.load('./npy/euclidean_sigmoid_hidden-size_1_test_acc.npy')

euclidean_relu_hidden_size_2_train_loss = np.load('./npy/euclidean_relu_hidden-size_2_train_loss.npy')
euclidean_sigmoid_hidden_size_2_train_loss = np.load('./npy/euclidean_sigmoid_hidden-size_2_train_loss.npy')
softmax_sigmoid_hidden_size_2_train_loss = np.load('./npy/softmax_sigmoid_hidden-size_2_train_loss.npy')

print(shorten(euclidean_sigmoid_hidden_size_1_train_loss,500))
print(shorten(softmax_sigmoid_hidden_size_1_train_loss,500))

def drawloss():
	plt.plot(range(1,60000,300),shorten(euclidean_relu_hidden_size_1_train_loss,300),linewidth=1.5,label='1-layer euclidean relu')
	plt.plot(range(1,60000,300),shorten(euclidean_sigmoid_hidden_size_1_train_loss,300),linewidth=1.5,label='1-layer euclidean sigmoid')
	plt.plot(range(1,60000,300),shorten(softmax_sigmoid_hidden_size_1_train_loss,300),linewidth=1.5,label='1-layer softmax sigmoid')

	plt.plot(range(1,60000,300),shorten(euclidean_relu_hidden_size_2_train_loss,300),color='red',linewidth=1.5,label='2-layer euclidean relu')
	plt.plot(range(1,60000,300),shorten(euclidean_sigmoid_hidden_size_2_train_loss,300),color='gray',linewidth=1.5,label='2-layer euclidean sigmoid')
	plt.plot(range(1,60000,300),shorten(softmax_sigmoid_hidden_size_2_train_loss,300),color='purple',linewidth=1.5,label='2-layer softmax sigmoid')
	plt.legend(loc='upper right')
	plt.xlabel("iterations")
	plt.ylabel("loss")
	plt.title("training loss")
	plt.savefig("loss.png")
	plt.show()

def drawacc():
	euclidean_relu_hidden_size_1_train_acc = np.load('./npy/euclidean_relu_hidden-size_1_train_acc.npy')
	euclidean_sigmoid_hidden_size_1_train_acc = np.load('./npy/euclidean_sigmoid_hidden-size_1_train_acc.npy')
	softmax_sigmoid_hidden_size_1_train_acc = np.load('./npy/softmax_sigmoid_hidden-size_1_train_acc.npy')
	euclidean_relu_hidden_size_2_train_acc = np.load('./npy/euclidean_relu_hidden-size_2_train_acc.npy')
	euclidean_sigmoid_hidden_size_2_train_acc = np.load('./npy/euclidean_sigmoid_hidden-size_2_train_acc.npy')
	softmax_sigmoid_hidden_size_2_train_acc = np.load('./npy/softmax_sigmoid_hidden-size_2_train_acc.npy')
	plt.plot(range(1,60000,300),shorten(euclidean_relu_hidden_size_1_train_acc,300),color='orange',linewidth=1.5,label='1-layer euclidean relu')
	plt.plot(range(1,60000,300),shorten(euclidean_sigmoid_hidden_size_1_train_acc,300),color='blue',linewidth=1.5,label='1-layer euclidean sigmoid')
	plt.plot(range(1,60000,300),shorten(softmax_sigmoid_hidden_size_1_train_acc,300),color='green',linewidth=1.5,label='1-layer softmax sigmoid')
	plt.plot(range(1,60000,300),shorten(euclidean_relu_hidden_size_2_train_acc,300),color='red',linewidth=1.5,label='2-layer euclidean relu')
	plt.plot(range(1,60000,300),shorten(euclidean_sigmoid_hidden_size_2_train_acc,300),color='gray',linewidth=1.5,label='2-layer euclidean sigmoid')
	plt.plot(range(1,60000,300),shorten(softmax_sigmoid_hidden_size_2_train_acc,300),color='purple',linewidth=1.5,label='2-layer softmax sigmoid')
	plt.legend(loc='lower right')
	plt.xlabel("iterations")
	plt.ylabel("accuracy")
	plt.title("training accuracy")
	plt.savefig("acc.png")
	plt.show()

drawacc()
