#!/usr/bin/python
import david as d
import random

def gen_data(xd = 3, size = 10):
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

net = d.NeuNetwork([3,4,1])
net.work(gen_data(),1,2,1)
