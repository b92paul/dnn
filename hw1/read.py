import numpy as np

train_set = "../../data/fbank/train.ark"
train_label = "../../data/label/train.lab"

def read_data(filename, cut=2000):
    Id = []
    res = []
    f = open(filename,'r')
    for i, line in enumerate(f):
        if i == cut:
            break
        if i%100000 == 0:
            print "read data %d" % i
        tmp = line.strip().split(' ')
        dt = np.array(tmp[1:],dtype=float)
        Id.append(tmp[0])
        res.append(dt)
    return Id,np.array(res)
Id,res =read_data(train_set,2000)

def read_label(filename, cut = 2000):
    lmap = {}
    m39_idx = {}
    count = 0
    f = open(filename,'r')
    for i, line in enumerate(f):
        if i% 100000 == 0:
            print "read label %d" % i
        if i == cut:
            break
        tmp = line.strip().split(',')
        lmap[tmp[0]] = tmp[1]
    return lmap

print res[3:20]
print Id[3:20]
mapping = {}
mapping["123"] = 4
mapping["2342342"] = 5
lmap = read_label(train_label,10000000)
print [lmap[x] for x in Id[3:2000]]
print lmap.has_key('asdf')

