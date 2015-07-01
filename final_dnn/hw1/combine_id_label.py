import sys

if len(sys.argv) != 4:
  print("label, id, mapfile")
  assert False

label_file = open(sys.argv[1])
id_file = open(sys.argv[2])
map_file = open(sys.argv[3])
lmap = {}
for line in map_file:
  tmp = line.strip().split()
  lmap[tmp[0]] = tmp[2]
print('Id,Prediction')
for line1, line2 in zip(id_file,label_file):
  name = line1.strip()
  label = lmap[line2.strip()]
  print(name+','+label)
