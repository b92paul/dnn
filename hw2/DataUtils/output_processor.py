import random
import numpy as np

from phone_mapper import Mapper

FBANK_FEATURES = 69
MFCC_FEATURES = 39
LABEL_COUNT = 48

mapper = Mapper()

def reduce(array):
	result = []
	for i in xrange(len(array)):
		if i == 0 or array[i] != array[i - 1]:
			result.append(array[i])
	if result[0] == 37:
		result = result[1:]
	if result[-1] == 37:
		result = result[:-1]
	return result

def to39(num):
	phone = mapper.get_phone(num, type="48")
	phone = mapper.ptrans(phone)
	phone = mapper.get_index(phone, type="48")
	return phone
def alpha(num):
	if num < 26:
		return chr(ord('a') + num)
	else:
		return chr(ord('A') + num - 26)

class FeatureProcessor:

	def __init__(self, path = '../../../data', limit = -1):
		#labels_48_file = open(path + '/label/train.lab', 'r')
		
		label_file = open(path + '/test_0.label', 'r')
		output_file = open(path + '/test_0.output', 'r')

		final_file = open(path + '/test_0.csv', 'w')

		self.labels = []


		self.read_label(label_file)
		self.read_and_output(output_file, final_file)

		print "Data Loaded"

	def read_label(self, file):
		for line in file:
			self.labels.append(line.split()[0])

	def read_and_output(self, file_in, file_out):
		file_out.write('id,phone_sequence\n')
		count = 0
		for line in file_in:
			file_out.write(self.labels[count] + ',')
			phones = map(int, line.split())
			phones = map(to39, phones)
			phones = reduce(phones)
			phones = ''.join(map(alpha, phones))
			file_out.write(phones +'\n')
			count += 1


def main():
	featureProc = FeatureProcessor()

if __name__ == '__main__':
	main()

