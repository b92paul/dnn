import os
test = open('test.txt','w')
testing_parse = open('testing_parse.txt','r');
test.write(str(5200)+'\n')
a = 0
for line in testing_parse:
    a=a+1
    array = line.split()
    test.write(str(len(array))+'\n')
    for i in xrange(len(array)):
        test.write(array[i]+'\n')
test.close()
testing_parse.close()
