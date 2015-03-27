import numpy as np

train_fbank = "../../data/fbank/train.ark"
train_mfcc = "../../data/mfcc/train.ark"
train_label = "../../data/label/train.lab"

test_fbank = "../../data/fbank/test.ark"
test_mfcc = "../../data/mfcc/test.ark"

train_merge = "../../data/merge/train.out"
label_merge = "../../data/merge/label.out"
label_map = "../../data/merge/lamp.out"
test_merge = "../../data/merge/test.out"
test_id = "../../data/merge/test_id.out"

check = 100000

def read_data(filename, cut=2000):
    res = {}
    f = open(filename,'r')
    for i, line in enumerate(f):
        if i == cut:
            break
        if i % check == 0:
            print "read " +filename+":%d" % i 
        tmp = line.strip().split(' ')
        dt = np.array(tmp[1:],dtype=float)
        res[tmp[0]] = dt
    print "read " +filename+ " done"
    return res

def read_label(filename, cut = 2000):
    id_label_mp = {}
    label_idx_mp = {}
    count = 0
    f = open(filename,'r')
    for i, line in enumerate(f):
        if i == cut:
            break
        tmp = line.strip().split(',')
        id_label_mp[tmp[0]] = tmp[1]
        if not label_idx_mp.has_key(tmp[1]):
            label_idx_mp[tmp[1]] = len(label_idx_mp)
    return id_label_mp, label_idx_mp
def idx2array(idx):
    res = np.zeros(48,dtype = float)
    res[idx] = 1
    return res

def merge_train(fbank, mfcc, ilm, lim):
    tr = open(train_merge,'w')
    lb = []
    for i, id in enumerate(fbank):
        if i % check == 0:
            print "merge train: %d" % i 
        idx = lim[ ilm[id] ]
        tr_tmp = np.concatenate((fbank[id],mfcc[id]))
        tr_tmp = ','.join(['%.8f' % num for num in tr_tmp])
        tr.write(tr_tmp+'\n')
        lb.append(idx2array(idx))
    # np.savetxt(train_merge, np.array(tr), delimiter=",")
    np.savetxt(label_merge, np.array(lb), delimiter=",", fmt='%.1f')

# merge train data
data_fbank = read_data(train_fbank,-1)
data_mfcc = read_data(train_mfcc,-1)
id_label_mp,label_idx_mp = read_label(train_label,-1)
merge_train(data_fbank, data_mfcc, id_label_mp, label_idx_mp)

# merge test data
data_fbank_t = read_data(test_fbank, -1)
data_mfcc_t  = read_data(test_mfcc, -1)
tm = []
ti = open(test_id,'w')
for id in data_fbank_t:
    tm.append(np.concatenate( (data_fbank_t[id], data_mfcc_t[id])) )
    ti.write(id+'\n')
np.savetxt(test_merge, tm, delimiter=",", fmt='%.8f')
