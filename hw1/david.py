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
            validation_data = training_data
        print "Time {0}: {1} / {2}".format(0, self.test(validation_data) , len(validation_data))

    def update_mini_batch(self, data, eta):
        new_bs = [np.zeros(b.shape) for b in self.biases]
        new_ws = [np.zeros(w.shape) for w in self.weights]
        for x,y in data:
            delta_bs,delta_ws = self.propagation(x,y)
            new_bs = [b+nb for b,nb in zip(new_bs,delta_bs)]
            new_ws = [w+nw for w,nw in zip(new_ws,delta_ws)]
        self.biases  = np.array([b-(eta/len(data))*nb for b, nb in zip(self.biases, new_bs)])
        self.weights = np.array([w-(eta/len(data))*nw for w, nw in zip(self.weights, new_ws)])

    def propagation(self, x, y):
        delta_biases,delta_weights = [np.zeros(b.shape) for b in self.biases],[np.zeros(w.shape) for w in self.weights]
        act = np.matrix(x).transpose()
        acts = [act]
        zs = [[]]
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,act)+b
            zs.append(z)
            act = sigmoid_vec(z)
            acts.append(act)
        delta = self.cost(acts[-1],y) * sigmoid_prime_vec(zs[-1])
        print zs[-1]
        delta_biases[-1] = delta
        delta_weights[-1] = np.dot(delta, acts[-2].transpose())
        for i in xrange(2, len(self.dimention)): # 2...n
            a=np.dot(self.weights[-i+1].transpose(), delta)
            b=sigmoid_prime_vec(zs[-i])
            delta = np.multiply(np.dot(self.weights[-i+1].transpose(), delta), sigmoid_prime_vec(zs[-i]))
            delta_biases[-i] = delta
            delta_weights[-i] = np.dot(delta, acts[-i-1].transpose())
        return delta_biases,delta_weights

    def feedforward(self,x):
        act = np.matrix(x).transpose()
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,act)+b
            act = sigmoid_vec(z)
        return act

    def test(self, data):
        for x,y in data:
            print "{0} <=> {1}".format(self.feedforward(x),y)
        return 0

    def cost(self, my_y, real_y):
        return my_y - real_y

    def split_validation(self, data):
        return data,[]

    def convert_to_tuple(self, data):
        #origin: [ [ x,x,x ], [y,y,y] ]
        #after: [(x,y),(x,y),(x,y)]
        return zip(data[0],data[1])


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