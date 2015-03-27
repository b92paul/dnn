#!/usr/bin/python
import david as d
import random
import numpy as np

def gen_data(xd = 20, size = 100):
    x,y= [],[]
    for i in xrange(size):
        c,s = [],0
        for j in xrange(xd):
            e = random.randint(0,1)
            c.append(e)
            s+=e
        if s>xd/2:
            s=1
        else:
            s=0
        x.append(c)
        y.append([s])
    return [x,y]

data = gen_data()
net = d.NeuNetwork([len(data[0][0]),2,1])
net.work(data,100,10,10)
