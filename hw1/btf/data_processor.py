from phone_mapper import Mapper

import random

class TrainDataProcessor:

	def __init__(self, limit = -1, valid_count = 200000):
		train_data_fbank = open('data/fbank/train.ark', 'r')
		train_data_mfcc = open('data/mfcc/train.ark', 'r')
		labels_48 = open('data/label/train.lab', 'r')
		# labels_1943 = open('state_label/train.lab', 'r')
		self.features = {}
		self.labels = {}
		self.valid_cursor = 0
		self.train_cursor = 0
		self.limit = limit
		self.record_list = []
		self.train_data_list = []
		self.valid_data_list = []
		self.mapper = Mapper()

		self.read_features(train_data_fbank)
		self.read_features(train_data_mfcc)
		self.read_labels(labels_48)

		self.valid_data_list = self.record_list[:valid_count]
		self.train_data_list = self.record_list[valid_count:]
		random.shuffle(self.valid_data_list)
		random.shuffle(self.train_data_list)

		print "Data Loaded"

	def read_features(self, file):
		CHECK_POINT = 100000
		count = 0
		for line in file:
			line = line.strip().split(' ')
			record_id = line[0]
			if record_id not in self.features:
				self.features[record_id] = []
				self.record_list.append(record_id)
			self.features[record_id].extend(map(float, line[1:]))
			count += 1
			if count == self.limit:
				break
			if count % CHECK_POINT == 0:
				print str(count) + ' entries loaded.'

	def read_labels(self, file):
		for line in file:
			line = line.strip().split(',')
			label = [0] * 48
			label[self.mapper.get_index(line[1])] = 1
			self.labels[line[0]] = label

	def count(self):
		return len(self.record_list)

	def get_features(self, record_id):
		return self.features[record_id]

	def get_labels(self, record_id):
		return self.labels[record_id]

	def get_train_record(self):

		ret = None
		if self.train_cursor >= len(self.train_data_list):
			random.shuffle(self.train_data_list)
			self.train_cursor = 0

		record_id = self.train_data_list[self.train_cursor]
		ret = (self.get_features(record_id), self.get_labels(record_id))
		self.train_cursor += 1
		return ret

	def get_valid_record(self):

		ret = None
		if self.valid_cursor >= len(self.valid_data_list):
			random.shuffle(self.valid_data_list)
			self.valid_cursor = 0

		record_id = self.valid_data_list[self.valid_cursor]
		ret = (self.get_features(record_id), self.get_labels(record_id))
		self.valid_cursor += 1
		return ret


class TestDataProcessor:
	def __init__(self):
		test_data_fbank = open('data/fbank/test.ark', 'r')
		test_data_mfcc = open('data/mfcc/test.ark', 'r')
		self.features = {}
		self.record_list = []
		self.cursor = 0
		self.read_features(test_data_fbank)
		self.read_features(test_data_mfcc)

	def count(self):
		return len(self.record_list)

	def read_features(self, file):
		for line in file:
			line = line.strip().split(' ')
			record_id = line[0]
			if record_id not in self.features:
				self.features[record_id] = []
				self.record_list.append(record_id)
			self.features[record_id].extend(map(float, line[1:]))

	def get_features(self, record_id):
		return self.features[record_id]

	def get_record(self, cursor = None):
		if cursor is not None:
			self.cursor = cursor
		ret = None
		if self.cursor >= len(self.record_list):
			return None

		record_id = self.record_list[self.cursor]
		ret = self.get_features(record_id)
		self.cursor += 1
		return ret


def main():
	data = TrainDataProcessor()
	print data.count()
	test_data = TestDataProcessor()
	print test_data.count()

if __name__ == '__main__':
	main()

