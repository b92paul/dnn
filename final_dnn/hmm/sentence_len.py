test_id = open('../../../data/final/test_id.ark')
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
