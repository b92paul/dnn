import random
import numpy as np
import sys

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
	return result

def trim(array):
	if array[0] == 37:
		array = array[1:]
	if array[-1] == 37:
		array = array[:-1]
	return array

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

def kill_single(array):
	ret = []
	for i in xrange(len(array)):
		if i > 1 and array[i] == array[i - 1] and array[i] == array[i - 2]:
			ret.append(array[i])
	return ret

class OutputProcessor:

	def __init__(self, path = '../../../data',  limit = -1, prefix='test_0'):
		#labels_48_file = open(path + '/label/train.lab', 'r')
		
		label_file = open(path + '/'+prefix+'.label', 'r')
		output_file = open(path + '/'+prefix+'.output', 'r')

		final_file = open(path + '/'+prefix+'.csv', 'w')

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
			phones = kill_single(phones)
			phones = map(to39, phones)
			phones = reduce(phones)
			phones = trim(phones)
			phones = ''.join(map(alpha, phones))
			file_out.write(phones +'\n')
			count += 1

def main():
        print "Args: ", str(sys.argv)
        if len(sys.argv) >= 2:
            outputProc = OutputProcessor(sys.argv[1],-1,sys.argv[2])
        else:
            outputProc = OutputProcessor()

if __name__ == '__main__':
	main()

