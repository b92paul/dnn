
from dnn import DeepNeuralNetwork
from data_processor import TrainDataProcessor

class Trainer:
	def __init__(self):

		self.model = DeepNeuralNetwork()
		self.model.init(input_count = 108,
										hidden_layer_numbers = [128],
										output_count = 48,
										learning_rate = 0.05)
		self.data = TrainDataProcessor(-1)

	def run(self):
		self.data.set_cursor(200000)
		count = 0
		while True:
			count += 1
			record = self.data.get_record()
			if record is None:
				break
			#print record
			self.model.forward(record[0])
			self.model.backpropagation(record[1])
			self.model.update()
			print str(count) + ' data entries processed.'

	def get_index(self, result):
		for i in xrange(len(result)):
			if result[i] == 1:
				return i
		return -1

	def validate(self):
		self.data.set_cursor(0)
		VALID_COUNT = 200000
		count = 0
		correct = 0
		while count < VALID_COUNT:
			count += 1
			record = self.data.get_record()
			if record is None:
				break
			#print record
			result = self.model.predict(record[0])
			answer = self.get_index(record[1])
			print '# ' + str(count) + ': ' + str(result) + ' ' + str(answer)
			if result == answer:
				correct += 1
		print 'validation: ' + str(float(correct) / float(VALID_COUNT))

	def save(self, filename = 'train.config'):
		self.model.save_model(filename)

	def load(self, filename = 'train.config'):
		self.model.load_model(filename)

def main():
	trainer = Trainer()
	EPOCH = 1
	trainer.load()
	for i in xrange(EPOCH):
		trainer.run()
	trainer.validate()
	trainer.save()

if __name__ == '__main__':
	main()
