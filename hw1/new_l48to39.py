#!/usr/bin/python
import numpy as np
train_label_path = "../../data/merge/label2.out"

mp_48_39_path = "../../data/phones/48_39.map"
mp_id_48_path = "../../data/merge/lmap2.out"

# out
mp_id_39_path = "../../data/merge/lmap_392.out"

train_label_39_path = "../../data/merge/label_392.out"


def gen39file(filename,outfile):
    f1 =open(mp_48_39_path,"r")
    f2 =open(mp_id_48_path,"r")
    m48_39 = {}
    mid_48 = {}
    m39_id = {}
    for line in f1:
        tmp = line.strip().split('\t')
        m48_39[tmp[0]]=tmp[1]
        if not m39_id.has_key(tmp[1]):
            m39_id[tmp[1]] = len(m39_id)
    for line in f2:
        tmp = line.strip().split(' ')
        mid_48[tmp[0]] = tmp[1]
    lb = []

    f5 = open(mp_id_39_path,"w")
    for idx in m39_id:
        print idx
        f5.write(str(m39_id[idx])+" "+idx+"\n")
    f5.close()

    f3 = open(filename,"r")
    f4 = open(outfile,"w")

    # read 48 [0,.,1] array map to 39 [0,0,..1] array
    for now,line in enumerate(f3):
        if now % 10000 ==0:
            print "at ",now," OAO"
        tmp = line.strip().split(',')
        tmp = np.array(tmp,dtype=float)
        idx = np.zeros(39,dtype=float)
        i = np.argmax(tmp)
        if tmp[i] > 0:
            oo = m39_id[m48_39[mid_48[str(i)]]]
            idx[oo] = 1
        lb.append(idx)
    f3.close()
    print "Write to out_file ~!"
    
    # outfile 39 array to outfile
    np.savetxt(f4, np.array(lb), delimiter=",", fmt='%.1f')
    f4.close()

        
gen39file(train_label_path, train_label_39_path)

