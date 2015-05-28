import sys
import random
doc_num = len(sys.argv)-1
def file2list(filename):
	f = open(filename)
	x = []
	for line in f:
		x.append(line)
	return x
y = []
for i in range(doc_num):
	y.append(file2list(sys.argv[i+1]))
f = open('merge_ans.csv','w')
for i in range(len(y[0])):
	if y[1][i] == y[2][i]:
		f.write(y[1][i])
	elif y[0][i] == y[1][i]:
		f.write(y[0][i])
	elif y[0][i] == y[2][i]:
		f.write(y[0][i])
	else:
		f.write(y[random.randint(0,2)][i])
