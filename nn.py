import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import os

size = 28
num_inter = 1568
num_output = 10
batch_size = 100
num_traindata = 50000
num_testdata = 10000
rate = 0.01

num_epoch = 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def softmax(x):
    alpha = np.max(x, axis=0, keepdims=False)
    return np.exp(x - alpha) / np.sum(np.exp(x - alpha), axis = 0, keepdims=True)
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred)) / batch_size
def categorical(y_true):
    category = np.zeros(num_output)
    category[y_true] = 1
    return category


mndata = MNIST("/export/home/imel1/takahashi/le4nn")
X, Y = mndata.load_training()
X = np.array(X)
Y = np.array(Y)

def test(W1,W2,b1,b2,batch):

    train_X = (X[batch] / 255).T
    train_Y = np.array([categorical(Y[i]) for i in batch]).T
    inter = sigmoid(np.dot(W1, train_X) + b1)
    output = softmax(np.dot(W2, inter) + b2)
    outnum = np.argmax(output, axis=0)
    gt = Y[batch]
    precision = np.where(outnum == gt, 1, 0)
    return np.sum(precision,dtype=float) / 1000

def train():

    #random.seed(0)

    W1 = np.random.normal(0, 1, (size * size, num_inter)).T
    W2 = np.random.normal(0, 1, (num_inter, num_output)).T
    b1 = np.random.normal(0, 1, (1, num_inter)).T
    b2 = np.random.normal(0, 1, (1, num_output)).T

    if os.path.isfile("W1.npy"):
        if np.load("W1.npy").shape[0] == num_inter:
            print("loading pre_trained weights...")
            W1 = np.load("W1.npy")
            W2 = np.load("W2.npy")
            b1 = np.load("b1.npy")
            b2 = np.load("b2.npy")
    #for i in [1]:
    #    for i in range(10):
    for epoch in range(num_epoch):
        for t in range(num_traindata / batch_size):

            batch = np.random.choice(range(num_traindata), batch_size)
            train_X = (X[batch] / 255).T
            train_Y = np.array([categorical(Y[i]) for i in batch]).T
            inter = sigmoid(np.dot(W1, train_X) + b1)
            output = softmax(np.dot(W2, inter) + b2)
            #print(output)
            #outnum = np.argmax(output, axis=0)
            loss = cross_entropy(train_Y, output)

            dev_entropy = (output - train_Y) / batch_size
            dev_inter = np.dot(W2.T, dev_entropy)
            dev_W2 = np.dot(dev_entropy, inter.T)
            dev_b2 = np.sum(dev_entropy, axis=1, keepdims=True)
            dev_inter_sigmoid = dev_inter * (1 - dev_inter)
            #dev_input = np.dot(W1.T, dev_inter_sigmoid)
            dev_W1 = np.dot(dev_inter_sigmoid, train_X.T)
            dev_b1 = np.sum(dev_inter_sigmoid, axis=1, keepdims=True)
            W1 -= rate * dev_W1
            W2 -= rate * dev_W2
            b1 -= rate * dev_b1
            b2 -= rate * dev_b2

        print("epoch" + str(epoch + 1) + " result")
        print ("loss",loss)
        print("acc", test(W1, W2, b1, b2, np.random.choice(range(num_traindata), 1000)))
        print("val_acc", test(W1, W2, b1, b2, np.random.choice(range(60000 - num_testdata, 60000), 1000)))
        print("")

    np.save("W1.npy",W1)
    np.save("W2.npy",W2)
    np.save("b1.npy",b1)
    np.save("b2.npy",b2)


input_train = input("train 0 : ")
if input_train == 0:
    train()
else:
    print(test(np.load("W1.npy"),np.load("W2.npy"),np.load("b1.npy"),np.load("b2.npy")))

