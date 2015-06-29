import sys
if len(sys.argv)!=3:
  assert False
print(sys.argv[1])
f = open(sys.argv[1])
fout = open(sys.argv[2],'w')
word = set(["fur","'em","ill","sin","kin","tim","tin","ski","al","x","ole","pa","gin","non-","knick-","y'all","se","o'","awe","ya","clot"])
yo =0
for line in f:
  pre = ""
  first = True
  tmp = line.strip().split()
  for w in tmp:
    if w not in word and w!=pre:
      pre = w
      if not first:
        fout.write(' ')
      else:
        first = False
      fout.write(w)
    else:
      yo+=1
  if first and len(tmp)>0:
    pass
    #fout.write('QQ')
    #assert False
  else:
    fout.write('\n')
print(yo)

