#!/usr/bin/python
import david as d
import random
import numpy as np

def gen_data(xd = 5, size = 10):
    x,y= [],[]
    for i in xrange(size):
        c,s = [],0
        for j in xrange(xd):
            e = random.randint(0,1)
            c.append(e)
            s+=e
        s = s%2
        x.append(c)
        y.append([s])
    return [x,y]

data = gen_data()
net = d.NeuNetwork([len(data[0][0]),3,1])
net.work(data,1,2,20)
