
import numpy

import math

import time

# DNN Layer
class Layer:
	def __init__(self, in_count, out_count, weight = None, bias = None, activation = numpy.tanh):
		weight = weight or (2 * numpy.random.sample((out_count, in_count)) - 1)
		self.weight = weight
		#self.weight = theano.shared(value = weight)
		self.bias = bias or (2 * numpy.random.sample(out_count) - 1)

		self.activation = activation 
		self.in_count = in_count
		self.out_count = out_count

	def set_in_neurons(self, in_values):
		self.in_values = in_values
		self.forward()

	def get_out_neurons(self):
		return self.out_values

	def forward(self):
		self.raw_out_values = numpy.dot(self.weight, self.in_values) + self.bias
		self.out_values = self.activation(self.raw_out_values)
		return self.out_values

	def update(self, learning_rate):
		self.weight -= self.grad * learning_rate
		self.bias -= self.delta * learning_rate

	def backpropagation(self, previous):
		self.delta = (1 - numpy.tanh(numpy.tanh(self.raw_out_values))) * previous
		self.grad = numpy.repeat([self.in_values], self.out_count, axis=0) * numpy.repeat(numpy.transpose([self.delta]), self.in_count, axis=1)
		
	def get_delta_weight(self):
		return numpy.dot(numpy.transpose(self.weight), self.delta)

	def set_weight(self, weight):
		self.weight = weight

	def set_bias(self, bias):
		self.bias = bias

# DNN
class DeepNeuralNetwork:
	def __init__(self):
		pass

	def init(self, input_count, hidden_layer_numbers, output_count, learning_rate):
		""" init with parameters

			input: input count
			layer_cou: 
			output: output
		"""
		self.input_count = input_count
		self.output_count = output_count
		self.learning_rate = learning_rate # learning rate
		self.hidden_layer_numbers = hidden_layer_numbers

		# Init layers
		self.layers = []
		hidden_layer_numbers.insert(0, input_count)
		hidden_layer_numbers.append(output_count)
		self.layer_count = len(hidden_layer_numbers) - 1

		for i in xrange(self.layer_count):
			self.layers.append(Layer(hidden_layer_numbers[i], hidden_layer_numbers[i + 1]))

	def update(self):
		for i in xrange(self.layer_count):
			self.layers[i].update(self.learning_rate)

	def forward(self, X):
		for i in xrange(self.layer_count):
			if i > 0:
				self.layers[i].set_in_neurons(self.layers[i - 1].get_out_neurons())
			else:
				self.layers[i].set_in_neurons(X)
		self.input = X
		self.output = self.layers[-1].get_out_neurons()
		return self.output

	def calculate_error(self, Y):
		return numpy.linalg.norm(self.output - Y)

	def backpropagation(self, Y):
		for i in xrange(self.layer_count - 1, -1, -1):
			if i == self.layer_count - 1:
				self.layers[i].backpropagation(2 * (self.layers[i].get_out_neurons() - Y))
			else:
				self.layers[i].backpropagation(self.layers[i + 1].get_delta_weight())
		
	def predict(self, X):
		self.forward(X)
		max_value = -100.0
		max_idx = -1
		output = self.layers[-1].get_out_neurons().tolist()
		for i in xrange(len(output)):
			if output[i] > max_value:
				max_value = output[i]
				max_idx = i
		return max_idx
		# TODO: predict one answer

	def save_model(self, filename = None):
		if filename is None:
			filename = time.strftime("%Y%m%d%H%M%S.config", time.gmtime())
		config = open(filename, 'w')
		config.write(str(self.input_count) + '\n')
		config.write(str(self.output_count) + '\n')
		config.write(' '.join(map(str, self.hidden_layer_numbers)) + '\n')
		config.write(str(self.learning_rate) + '\n')
		for i in xrange(self.layer_count):
			temp_array = self.layers[i].weight
			for j in xrange(self.layers[i].out_count):
				config.write(' '.join(map(str, temp_array[j])) + '\n')
			config.write(' '.join(map(str, self.layers[i].bias)) + '\n')
		config.close()

	def load_model(self, filename):
		config = open(filename, 'r')
		self.input_count = int(config.readline())
		self.output_count = int(config.readline())
		self.hidden_layer_numbers = map(int, config.readline().split(' '))
		self.learning_rate = float(config.readline())
		self.layer_count = len(self.hidden_layer_numbers) - 1

		self.layers = []
		for i in xrange(self.layer_count):
			new_layer = Layer(self.hidden_layer_numbers[i], self.hidden_layer_numbers[i + 1])
			weight = []
			for j in xrange(self.hidden_layer_numbers[i + 1]):
				weight.append(map(float, config.readline().split(' ')))
			bias = map(float, config.readline().split(' '))
			new_layer.set_weight(weight)
			new_layer.set_bias(bias)
			self.layers.append(new_layer)


def test():
	dnn = DeepNeuralNetwork()
	dnn.init(input_count = 3, hidden_layer_numbers = [5, 8, 6], output_count = 1, learning_rate = 0.1)

	train_data_X = [[1, 0, 0], # 0
								[1, 1, 0], # 1
								[0, 0, 0], # 0
								[1, 0, 0], # 0
								[0, 1, 0], # 1
								[1, 0, 1], # 1
								[0, 0, 0], # 0
								[1, 0, 0], # 0
								[0, 1, 0], # 1
								[1, 0, 1]] # 1

	train_data_Y = [[0], [1], [0], [0], [1],
									[1], [0], [0], [1], [1]]

	t0 = time.time()

	#dnn.load_model('20150328131743.config')

	for a in xrange(2000):
		for i in xrange(len(train_data_X)):
			ret = dnn.forward(train_data_X[i])
			#print dnn.calculate_error(train_data_Y[i])
			dnn.backpropagation(train_data_Y[i])
			dnn.update()

	for i in xrange(len(train_data_X)):
		ret = dnn.predict(train_data_X[i])
		print ret
		print train_data_Y[i]
	
	print 'Time Elapsed:'
	print time.time() - t0
	#dnn.save_model()

def main():
	test()
	# run()

if __name__ == "__main__":
	main()
