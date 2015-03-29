import numpy as np
train_label_path = "../../data/merge/label.out"
test_label_path = "../../data/merge/test.out"
mp_48_39_path = "../../data/phones/48_39.map"
mp_id_48_path = "../../data/merge/lmap.out"
mp_id_39_path = "../../data/merge/lmap_39.out"

train_label_39_path = "../../data/merge/labal_39.out"
test_label_39_path = "../../data/merge/test_39.out"

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
        print tmp
        mid_48[tmp[0]] = tmp[1]
    print mid_48
    print m48_39
    print m39_id
    lb = []
    f3 = open(filename,"r")
    f4 = open(outfile,"w")
    for line in f3:
        tmp = line.strip().split(',')
        tmp = np.array(tmp,dtype=float)
        for i in xrange(len(tmp)):
            idx = np.zeros(39,dtype=float)
            if tmp[i] > 0:
                oo = m39_id[m48_39[mid_48[str(i)]]]
                idx[oo] = 1
            lb.append(idx)
    np.savetxt(out_file, np.array(lb), delimiter=",", fmt='%.1f')

    
gen39file(train_label_path, train_label_39_path)

