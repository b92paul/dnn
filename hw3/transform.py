import os
novel_vector = open('line_number.txt','w')
valid = open('valid.txt','w')
train = open("training2.txt","r")
novel_vector.write(str(2400000)+'\n')
valid.write(str(400000)+'\n')
a = 0
for line in train:
    a = a+1
    array = line.split()
    if a <= 2400000:
        novel_vector.write(str(len(array))+'\n')
        for i in xrange(len(array)):
            novel_vector.write(array[i]+'\n')
    elif a<= 2800000:
        valid.write(str(len(array))+'\n')
        for i in xrange(len(array)):
            valid.write(array[i]+'\n')
train.close()
valid.close()
novel_vector.close()
