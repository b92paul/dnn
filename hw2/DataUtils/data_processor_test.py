import random
import numpy as np

from phone_mapper import Mapper

FBANK_FEATURES = 69
MFCC_FEATURES = 39
LABEL_COUNT = 48

class FeatureProcessor:

	def __init__(self, path = '../../../data', limit = -1):
		#labels_48_file = open(path + '/label/train.lab', 'r')
		fbank_features_file = open(path + '/fbank/test.ark', 'r')
		
		output_file = open(path + '/test_0.ark', 'w')
		self.LIMIT = limit
		self.features = {}
		self.labels = {}
		self.record_list = []
		self.mapper = Mapper()

		#self.read_label(labels_48_file)
		self.read_feature(fbank_features_file)
		self.output(output_file)

		print "Data Loaded"

	def read_feature(self, file):
		CHECK_POINT = 100000
		self.count = 0
		for line in file:
			line = line.strip().split()
			label = line[0].split('_')
			label = '_'.join(label[:2])

			if label not in self.labels:
				self.labels[label] = []
				self.features[label] = []
				self.record_list.append(label)

			self.features[label].append(map(float, line[1:]))
			self.count += 1
			if self.count == self.LIMIT:
				break
			if self.count % CHECK_POINT == 0:
				print str(self.count) + ' entries loaded.'

	def output(self, file):
		self.record_list.sort()
		file.write(str(len(self.record_list)) + '\n')
		for label in self.record_list:
			frames = self.features[label]
			labels = [0] * len(frames)
			file.write(str(len(frames)) + ' ' + str(len(frames[0])) + '\n')
			file.write(label + '\n')
			for frame in frames:
				file.write(' '.join(map(str, frame)) + '\n')
			file.write(' '.join(map(str, labels)) + '\n')

def main():
	featureProc = FeatureProcessor()

if __name__ == '__main__':
	main()

