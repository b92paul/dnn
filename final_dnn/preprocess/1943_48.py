dir_path = '../../data/final/'
map_file = open(dir_path + 'state_48_39.map')
f48_id = open(dir_path + '48_idx_chr.map')
f1943_file = open(dir_path + 'f_1943.lab')
map_48_id = {}
for line in f48_id:
  tmp = line.strip().split()
  map_48_id[tmp[0]] = tmp[1]

state_48_map = {}

fout = open(dir_path + "f_48.lab",'w')

for line in map_file:
  tmp = line.strip().split()
  state_48_map[ tmp[0] ] = tmp[1]

for line in f1943_file:
  tmp = map_48_id[ state_48_map[line.strip()] ]
  fout.write(tmp+'\n')

