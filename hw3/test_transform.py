import os
vector_path = 'word2vec-read-only/novel.txt'
vector_file = open(vector_path,"r")
number = 0
vector_len = 0
dictionary = {}
for i,line in enumerate(vector_file):
    if i == 0:
        number,vector_len = [ int(x) for x in line.split()]
        print number,vector_len
    else:
        array = []
        key = ''
        for j,x in enumerate(line.split()):
            if j == 0:
                key = x
            else:
                array.append(float(x))
        dictionary[key] = array
vector_file.close()
print "OK"
test = open('test.txt','w')
testing_parse = open('testing_parse.txt','r');
test.write(str(5200)+'\n')
a = 0
for line in testing_parse:
    a=a+1
    array = line.split()
    test.write(str(len(array))+'\n')
    for i in xrange(len(array)):
        for j in xrange(200):
            if array[i] in dictionary:
                test.write(str(dictionary[array[i]][j])+'\n')
            else:
                test.write("0\n")
test.close()
testing_parse.close()
