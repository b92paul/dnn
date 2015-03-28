#!/usr/bin/python
import david as d
import random
import numpy as np

def gen_data(xd = 10, size = 200):
    x,y= [],[]
    for i in xrange(size):
        c,s = [],0
        for j in xrange(xd):
            e = random.randint(0,1)
            c.append([e])
            s+=e
        if s>xd/2:
            y.append(np.array([[0],[1]]))
        else:
            y.append(np.array([[1],[0]]))
        x.append(np.array(c))
    #print [x,y]
    return [x,y]

data = gen_data()
'''
net = d.NeuNetwork([10,2,2])
net.work(data,200,10,0.5)
exit()
'''

def read_x(cut=100000):
    ret = []
    f=open('../../data/merge/f_train.out', 'r')
    for i, line in enumerate(f):
        if i==cut:
            break;
        if i%10000==0:
            print i
        tmp = line.split(',')
        ret.append(np.array([tmp],dtype=float).transpose())
    return ret

def read_y(cut=100000):
    ret = []
    f=open('../../data/merge/label.out', 'r')
    for i, line in enumerate(f):
        if i==cut:
            break;
        if i%10000==0:
            print i
        tmp = line.split(',')
        ret.append(np.array([tmp],dtype=float).transpose())
    return ret

def work_speech():
    x = read_x()
    y = read_y()
    net = d.NeuNetwork([69, 28, 58, 48])
    net.work([x,y], 50, 500, 1.0)

work_speech()