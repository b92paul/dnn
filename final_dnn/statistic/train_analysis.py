import math
import sys
dir_path = "../../data/final/"
train_file = open(dir_path + "f_train.ark")
norm_file = open("preprocess/res.out","w")
avg = [0.0 for _ in range(69)]
res = [0.0 for _ in range(69)]
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
  count += 1
if w:
  for a, r in zip(avg,res):
    norm_file.write("%.8f %.8f\n" % (a/count,math.sqrt(r/count)) )
else:
  for a, r in zip(avg,res):
    print("%.8f %.8f" % (a/count,math.sqrt(r/count)) )
