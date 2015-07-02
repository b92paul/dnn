import sys
if len(sys.argv)!=2:
  print("input filename!")
  assert False
# test_id = open('../../../data/final/test_id_hw1.ark')
test_id = open(sys.argv[1])
total = 0
pre = ""
count = 0
for line in test_id:
  tmp = line.strip().split('_')
  name = tmp[0]+'_'+tmp[1]
  
  if name != pre:
    if pre!="":
      print(pre,count)
      total+=count
    count = 1
    pre = name
  else:
    count += 1
print(pre,count)
total+=count
