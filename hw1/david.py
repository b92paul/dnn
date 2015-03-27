import numpy as np
import random
import theano
import theano.tensor as T

class NeuNetwork():
    def __init__(self, dimention):
        self.dimention = dimention
        self.biases = [np.random.randn(y, 1) for y in dimention[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(dimention[:-1], dimention[1:])]
        print self.biases
        print self.weights

    def work(self, data, train_count, mini_batch_size, eta, test_data=None):
        data = self.convert_to_tuple(data)
        training_data,validation_data = self.split_validation(data)
        self.SGD(training_data, train_count, mini_batch_size, eta, test_data)

    def SGD(self, training_data, train_count, mini_batch_size, eta, test_data=None):
        print training_data
        for t in xrange(train_count):
            batches, ebp, total_size = [], 0, len(training_data)
            while ebp < total_size:
                batches.append(training_data[ebp:ebp+mini_batch_size])
                ebp += mini_batch_size
            for batch in batches:
                self.update_mini_batch(batch, eta)

    def update_mini_batch(self, data, eta):
        print data

    def split_validation(self, data):
        return data,[]

    def convert_to_tuple(self, data):
        #origin: [ [ x,x,x ], [y,y,y] ]
        #after: [(x,y),(x,y),(x,y)]
        ret = []
        for i in xrange(len(data[0])):
            ret.append((data[0][i],data[1][i]))
        random.shuffle(ret) 
        return ret


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

'''
import david as d
net = d.NeuNetwork([3,4,1])
net.work([[[1,1,1],[1,0,1]],[[1],[0]]],1,1,1)
'''