from math import log
xy_file = open("test_prob.out")
yy_file = open("hmm_table.out")

yy_table = []
for line in yy_file:
    yy_table.append( [float(x) for x in line.strip().split()] )
test_num = int(xy_file.next())
DP = [0.0 for _ in range(48)]
DPstr = [[i] for i in range(48)]
ans = open("out/48_nn_hmm.ark","w")
def argmax(x):
    idx = -1
    for i,y in enumerate(x):
        if idx == -1:
            idx = i
        elif x[i]>x[idx]:
            idx = i
    return idx
for _ in range(test_num):
    name = xy_file.next()
    size = int(xy_file.next())
    print name,size
    for i in range(size):
        line = xy_file.next().strip().split()
        prob = [log(float(x)+1e-10) for x in line ]
        if i == 0:
            DP = prob
            DPstr = [[i] for i in range(48)]
        else:
            nDP = [-1e30 for _ in range(48)]
            nDPstr = []
            for j in range(48):
                m_idx = -1
                for k in range(48):
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
