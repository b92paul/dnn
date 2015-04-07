import numpy as np

#train file paths
train_fbank = "../../data/fbank/train.ark"
train_mfcc = "../../data/mfcc/train.ark"
train_label = "../../data/label/train.lab"

# test file paths
test_fbank = "../../data/fbank/test.ark"
test_mfcc = "../../data/mfcc/test.ark"

# merge file paths
train_merge = "../../data/merge/f_train2.out"
label_merge = "../../data/merge/label2.out"
label_map = "../../data/merge/lmap2.out"
test_merge = "../../data/merge/f_test2.out"
test_id = "../../data/merge/test_id2.out"

check = 100000

def read_data(filename, cut=2000):
    res = []
    f = open(filename,'r')
    for i, line in enumerate(f):
        if i == cut:
            break
        if i % check == 0:
            print "read " +filename+":%d" % i 
        tmp = line.strip().split(' ')
        dt = np.array(tmp[1:],dtype=float)
        res.append((tmp[0], dt))
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

def merge_train(fbank, ilm, lim):
    tr = open(train_merge,'w')
    lb = []
    for i, (x,y) in enumerate(fbank):
        if i % check == 0:
            print "merge train: %d" % i 
        idx = lim[ ilm[x] ]
        tr_tmp = y
        tr_tmp = ','.join(['%.8f' % num for num in tr_tmp])
        tr.write(tr_tmp+'\n')
        lb.append(idx2array(idx))
    np.savetxt(label_merge, np.array(lb), delimiter=",", fmt='%.1f')

def merge_test(fbank):
    tm = []
    ti = open(test_id,'w')
    for (x,y) in fbank:
        tm.append( y )
        ti.write(x+'\n')
    np.savetxt(test_merge, tm, delimiter=",", fmt='%.8f')
    ti.close()


# read train label and create mapping table
id_label_mp,label_idx_mp = read_label(train_label,-1)

# read train data
data_fbank = read_data(train_fbank,-1)

# merge train data
merge_train(data_fbank, id_label_mp, label_idx_mp)

# read test data
data_fbank_t = read_data(test_fbank, -1)

# merge test data
merge_test(data_fbank_t)

f_lmap = open(label_map,'w')
for label in label_idx_mp:
    f_lmap.write(str(label_idx_mp[label]) + ' ' + label+ '\n')