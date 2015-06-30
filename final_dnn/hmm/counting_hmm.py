from math import log
f = open("../../../data/final/f_1943.lab")

label_num = 1943

last = -1
table = [[0.0 for _ in range(label_num)] for _ in range(label_num)]
p1 = [0.0 for _ in range(label_num) ]
#print(table)
total = 0.0
for line in f:
    total += 1
    now = int(line.strip())
    p1[now] += 1
    if last != -1:
      table[now][last] += 1
    last = now
for i in range(label_num):
    p1[i] /= total
    for j in range(label_num):
        table[i][j]/=(total-1)
p1p2 = [[0 for _ in range(label_num)] for _ in range(label_num)]
for i in range(label_num):
    for j in range(label_num):
        p1p2[i][j] = log( (table[i][j])/p1[i]+ 1e-8)
hmm_table = open("hmm/hmm_table.out","w")
for i in range(label_num):
    for j in range(label_num):
        hmm_table.write("%.8f " % p1p2[i][j])
    hmm_table.write("\n")
hmm_table.close()
f.close()
print(sum(p1))
