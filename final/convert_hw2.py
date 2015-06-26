import sys
lmap_file = open('../../data/48_idx_chr.map')
lmap = {}
for line in lmap_file:
    tmp = line.strip().split()
    lmap[tmp[2]] = tmp[0]
if len(sys.argv) <=2:
    print('python3 converti_hw2.py file_in file_out!')
else:
    fin = open(sys.argv[1])
    fout = open(sys.argv[2],'w')
    for i,line in enumerate(fin):
        if i == 0:
            print(line.strip())
        else:
            tmp = line.strip().split(',')
            for j,c in enumerate(tmp[1]):
								fout.write(lmap[c])
								if j!=len(tmp[1])-1:
										fout.write(' ')
								else:
										fout.write('\n')
            print(tmp[0],tmp[1])
    fout.close()
