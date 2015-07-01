from math import log
xy_file = open("test_prob.out")
yy_file = open("hmm_table.out")
name_file = open("sentence_name_len.out")
yy_table = []
for line in yy_file:
    yy_table.append( [float(x) for x in line.strip().split()] )
test_num = 0
name_list = []
len_list = []
for line in name_file:
  tmp = line.strip().split()
  name_list.append(tmp[0])
  len_list.append(int(tmp[1]))

DP = [0.0 for _ in range(1943)]
DPstr = [[i] for i in range(1943)]
ans = open("out/1943_nn_hmm.ark","w")
def argmax(x):
    idx = -1
    for i,y in enumerate(x):
        if idx == -1:
            idx = i
        elif x[i]>x[idx]:
            idx = i
    return idx
for name, size in zip(name_list, len_list):
    print name,size
    for i in range(size):
        line = xy_file.next().strip().split()
        prob = [log(float(x)+1e-10) for x in line ]
        if i == 0:
            DP = prob
            DPstr = [[i] for i in range(1943)]
        else:
            nDP = [-1e30 for _ in range(1943)]
            nDPstr = []
            for j in range(1943):
                m_idx = -1
                for k in range(1943):
                    if DP[k] + yy_table[k][j] > nDP[j]:
                        m_idx = k
                        nDP[j] = DP[k] + yy_table[k][j]
                nDP[j] += prob[j]
                nDPstr.append(DPstr[m_idx]+[j])
            DP = nDP
            DPstr = nDPstr
    ans_str = DPstr[argmax(DP)]
    ans.write(name)
    for x in ans_str:
        ans.write("%d " % x)
    ans.write("\n")
ans.close()
