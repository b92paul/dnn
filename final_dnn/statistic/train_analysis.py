import math
import sys
dir_path = "../../data/final/"
train_file = open(dir_path + "f_train.ark")
norm_file = open("preprocess/res.out","w")
avg = [0.0 for _ in range(69)]
res = [0.0 for _ in range(69)]
Min = [1e6 for _ in range(69)]
Max = [-1e6 for _ in range(69)]
count = 0.0
cut = -1
w = True
if len(sys.argv) >= 2:
  cut = int(sys.argv[1])
if len(sys.argv) == 3:
  train_file = open(sys.argv[2])
  w = False
for c,line in enumerate(train_file):
  if c == cut:
    break
  if (c+1) % 100000 ==0:
    print("read",c+1)
  s = 0.0  
  tmp = line.strip().split()
  for i,x in enumerate(tmp):
    x = float(x)
    res[i] += x**2
    avg[i] += x
    Min[i] = min(Min[i], x)
    Max[i] = max(Max[i], x)
  count += 1
if w:
  for a, r in zip(avg,res):
    norm_file.write("%.8f %.8f\n" % (a/count,math.sqrt(r/count)) )
else:
  for a, r, m, M in zip(avg,res,Min,Max):
    ex = a/count
    ex2 = r/count
    var = math.sqrt(ex2-ex**2)
    print("E[X]:%.8f Var(x):%.8f Min:%.8f Max:%.8f" % (a/count,var,m,M) )
