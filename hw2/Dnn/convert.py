name = "v4"
cut = 2
f = open("ans/48_"+name+".out","w")
dnn_res = open("out/48_"+name+".ark")
lmap39 = open("../../../data/merge/lmap2.out")
lab_map = {}
cmap = open("../../../data/48_idx_chr.map")
chr_map = {}

for line in lmap39:
    tmp = line.strip().split()
    lab_map[tmp[0]] = tmp[1]
for line in cmap:
    tmp = line.strip().split()
    chr_map[tmp[0]] = tmp[2]
print chr_map,lab_map

def strim(data, out_file):
    res = []
    idx = 0
    for x in xrange(len(data)):
        idx = idx + 1
        if x == len(data)-1 or data[x] != data[x+1]:
            if idx > cut:
                now = chr_map[lab_map[ data[x] ]]
                res.append(now)
            idx = 0
    string = ""
    for i in range(len(res)):
        if i == len(res)-1 or res[i] != res[i+1]:
            string = string + res[i]
    string = string.strip("L")
    out_file.write(string+"\n")

f.write("id,phone_sequence\n")
for line in dnn_res:
    name = line.strip()
    data = dnn_res.next().strip().split()
    f.write(name+ ',')
    strim(data, f)
f.close()
