import sys
f_name = open(sys.argv[1])
f_res = open(sys.argv[2])
f_out = open(sys.argv[3],'w')
f_chrmap = open('timit.chmap')
chrmap = {}
for i,line in enumerate(f_chrmap):
	tmp = line.strip().split()
	chrmap[tmp[0]] = tmp[1]
	# print(tmp[1])
name = []
res = []
title = next(f_name)
#print('title',title)
f_out.write('id,sequence\n')
for line1,line2 in zip(f_name,f_res):
	tmp = line1.strip().split(',')
	res = tmp[0]+','
	for word in line2.strip().split():
		res+=chrmap[word]
	res+='\n'
	#print(tmp[0])
	#print(res)
	f_out.write(res)
	#name.append(tmp[0])
f_name.close()
f_res.close()
f_out.close()
#for line in enumerate(f_res):
#	print(line)
