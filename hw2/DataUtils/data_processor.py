import random
import numpy as np

from phone_mapper import Mapper

FBANK_FEATURES = 69
MFCC_FEATURES = 39
LABEL_COUNT = 48

class FeatureProcessor:

	def __init__(self, path = '../../../data', limit = -1):
		labels_48_file = open(path + '/label/train.lab', 'r')
		fbank_features_file = open(path + '/fbank/train.ark', 'r')
		
		train_output_file = open(path + '/train_0.ark', 'w')
		valid_output_file = open(path + '/valid_0.ark', 'w')
		self.LIMIT = limit
		self.features = {}
		self.labels = {}
		self.record_list = []
		self.mapper = Mapper()

		self.read_label(labels_48_file)
		self.read_feature(fbank_features_file)
		record_count = len(self.record_list)
		train_count = record_count * 4 / 5
		valid_count = record_count - train_count
		random.shuffle(self.record_list)
		self.output(train_output_file, self.record_list[:train_count])
		self.output(valid_output_file, self.record_list[train_count:])

		print "Data Loaded"

	def read_feature(self, file):
		CHECK_POINT = 100000
		self.count = 0
		for line in file:
			line = line.strip().split()
			label = line[0].split('_')
			label = '_'.join(label[:2])

			self.features[label].append(map(float, line[1:]))
			self.count += 1
			if self.count == self.LIMIT:
				break
			if self.count % CHECK_POINT == 0:
				print str(self.count) + ' entries loaded.'

	def read_label(self, file):
		for line in file:
			line = line.strip().split(',')
			label = line[0].split('_')
			label = '_'.join(label[:2])

			phone = self.mapper.get_index(line[1])
			if label not in self.labels:
				self.labels[label] = []
				self.features[label] = []
				self.record_list.append(label)
			self.labels[label].append(phone)

	def output(self, file, records):
		records.sort()
		file.write(str(len(records)) + '\n')
		for label in records:
			frames = self.features[label]
			file.write(str(len(frames)) + ' ' + str(len(frames[0])) + '\n')
			file.write(label + '\n')
			for frame in frames:
				file.write(' '.join(map(str, frame)) + '\n')
			file.write(' '.join(map(str, self.labels[label])) + '\n')

def main():
	featureProc = FeatureProcessor()

if __name__ == '__main__':
	main()

