from math import log
f = open("../../../data/merge/label2.out")

last = -1
table = [[0.0 for _ in range(48)] for _ in range(48)]
p1 = [0.0 for _ in range(48) ]
print table
total = 0.0
for line in f:
    total += 1
    now = line.strip().split(',')
    for i,x in enumerate(now):
        if x == '1.0':
            p1[i] += 1
            if last != -1:
                table[i][last] += 1
            last = i
            break

for i in range(48):
    p1[i] /= total
    for j in range(48):
        table[i][j]/=(total-1)
p1p2 = [[0 for _ in range(48)] for _ in range(48)]
for i in range(48):
    for j in range(48):
        p1p2[i][j] = log( (table[i][j])/p1[i]+ 1e-8)
hmm_table = open("hmm_table.out","w")
for i in range(48):
    for j in range(48):
        hmm_table.write("%.8f " % p1p2[i][j])
    hmm_table.write("\n")
hmm_table.close()
f.close()
print p1
