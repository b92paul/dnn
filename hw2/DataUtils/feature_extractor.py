import random
import numpy as np

from phone_mapper import Mapper

FBANK_FEATURES = 69
MFCC_FEATURES = 39
LABEL_COUNT = 48

class FeatureExtractor:

	def __init__(self, path = '../../../data', limit = -1):
		fbank_features_file = open(path + '/faem0_si1392.ark', 'r')
		labels_48_file = open(path + '/faem0_si1392.lab', 'r')
		
		# labels_1943 = open('state_label/train.lab', 'r')
		self.features = {}
		self.labels = {}
		self.record_list = []
		self.mapper = Mapper()

		self.read_features(fbank_features_file)
		self.read_labels(labels_48_file)

		self.process_features()
		print "Data Loaded"

	def read_features(self, file):
		CHECK_POINT = 100000
		self.count = 0
		for line in file:
			line = line.strip().split()
			record_id = line[0]
			if record_id not in self.features:
				self.features[record_id] = []
				self.record_list.append(record_id)
				print record_id
			self.features[record_id].extend(map(float, line[1:]))
			self.count += 1
			if self.count % CHECK_POINT == 0:
				print str(count) + ' entries loaded.'

	def read_labels(self, file):
		for line in file:
			line = line.strip().split(',')
			#label = [0] * 48
			#label[self.mapper.get_index(line[1])] = 1
			self.labels[line[0]] = line[1]

	def process_features(self):
		# Calculate observation
		self.observation = np.zeros((48, 69))
		self.transition = np.zeros((48, 48))
		prev = None
		for entry in self.record_list:
			feature = self.features[entry]
			label = self.labels[entry]
			index = self.mapper.get_index(label)

			# Calculate observation
			self.observation[index] += np.asarray(feature)
			#print self.observation[index]
			# Calculate transition
			if prev is not None:
				self.transition[prev][index] += 1.0
			prev = index

	def output_features(self, path = '../../../data', file = '/faem0_si1392_psi.csv'):
		f = open(path + file, 'w')
		f.write("id,feature\n")
		for i in xrange(48):
			for j in xrange(69):
				f.write("faem0_si1392_" + str(i * 69 + j) + "," + "{:.20f}".format(self.observation[i][j]) + "\n")
		for i in xrange(48):
			for j in xrange(48):
				f.write("faem0_si1392_" + str(48 * 69 + i * 48 + j) + "," + "{:.20f}".format(self.transition[i][j]) + "\n")

	def get_features(self, record_id):
		return self.features[record_id]

	def get_labels(self, record_id):
		return self.labels[record_id]

	def output(self):
		print 'XD'

def main():
	extractor = FeatureExtractor()
	extractor.process_features()
	extractor.output_features()
	print extractor.count

if __name__ == '__main__':
	main()

