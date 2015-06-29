import sys
import math
f = open(sys.argv[1])
now = [[] for _ in range(535)]
idx = 0
pre = ""
wnum = 0.0
idf = {}
def add_idf(w,wnum):
  if w in idf:
    idf[w] += 1.0
  else:
    idf[w] = 1.0
  wnum += 1.0
  return wnum
def idf_num(w):
  return math.log(wnum/idf[w])

for line in f:
  tmp = line.strip()
  if tmp =="":
    if tmp != pre:
      idx += 1
  else:
    now[idx].append(tmp)
    for w in tmp.split():
      wnum = add_idf(w,wnum)
  pre = tmp
#print(wnum)
#print(idf_num('leadership'))
for l in now:
  w_count = {}
  for sen in l:
    tmp = sen.split()
    for w in tmp:
      if w in w_count:
        w_count[w] += len(w)**2#idf_num(w)
      else:
        w_count[w]  = len(w)**2#idf_num(w)
  res = []
  for sen in l:
    tmp = sen.split()
    score = 0
    for w in tmp:
      score += w_count[w] 
    score /= len(tmp)**2#math.sqrt(len(tmp))
    res.append((score,1000-len(tmp),sen))
    res.sort()
    res.reverse()
  #print(res[0][2])
  #for r in res:
  #  print(r)
  
  pre = ""
  num = 0
  for i,r in enumerate(res):
    if num > 30:
      print('')
      break
    if r[2] != pre:
      num+=1
      print(r[2])
      pre = r[2]
  
#print(idx)
