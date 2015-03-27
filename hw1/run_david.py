#!/usr/bin/python
import david as d
import random
import numpy as np


def gen_data(xd = 5, size = 100):
    x,y= [],[]
    for i in xrange(size):
        c,s = [],0
        for j in xrange(xd):
            e = random.randint(0,1)
            c.append(e)
            s+=e
        if s>xd/2:
            y.append([0,1])
        else:
            y.append([1,0])
        x.append(c)
    return [x,y]

data = gen_data()
net = d.NeuNetwork([len(data[0][0]),3,len(data[1][0])])
net.work(data,10,10,0.1)

exit()

def read_x(cut=5000):
    ret = []
    f=open('../../data/merge/train.out', 'r')
    for i, line in enumerate(f):
        if i==cut:
            break;
        if i%10000==0:
            print i
        tmp = line.split(',')
        ret.append(np.array(tmp,dtype=float))
    return ret

def read_y(cut=5000):
    ret = []
    f=open('../../data/merge/label.out', 'r')
    for i, line in enumerate(f):
        if i==cut:
            break;
        if i%10000==0:
            print i
        tmp = line.split(',')
        ret.append(np.array(tmp,dtype=float))
    return ret

def work_speech():
    x = read_x()
    y = read_y()
    net = d.NeuNetwork([108, 30, 30, 48])
    net.work([x,y], 30, 500, 0.1)

work_speech()