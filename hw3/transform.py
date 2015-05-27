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
novel_vector = open('novel_vector.txt','w')
test = open('test.txt','w')
train = open("training2.txt","r")
valid = open("valid.txt","w")
novel_vector.write(str(2400000)+'\n')
valid.write(str(498204)+'\n')
test.write(str(400000)+'\n')
print '2400000'
a = 0
for line in train:
    a=a+1
    if a%10000==0:
        print a,len(array)
    if a <= 400000:
        array = line.split()
        test.write(str(len(array))+'\n')
        for i in xrange(len(array)):
            for j in xrange(200):
                if array[i] in dictionary:
                    test.write(str(dictionary[array[i]][j])+'\n')
                else:
                    test.write("0\n")
    if a <= 2400000:
        array = line.split()
        novel_vector.write(str(len(array))+'\n')
        for i in xrange(len(array)):
            for j in xrange(200):
                if array[i] in dictionary:
                    novel_vector.write(str(dictionary[array[i]][j])+'\n')
                else:
                    novel_vector.write("0\n")
    else:
        array = line.split()
        valid.write(str(len(array))+'\n')
        for i in xrange(len(array)):
            for j in xrange(200):
                if array[i] in dictionary:
                    valid.write(str(dictionary[array[i]][j])+'\n')
                else:
                    valid.write("0\n")
train.close()
novel_vector.close()
