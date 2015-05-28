test_name = 'testing_data.txt'
test_parse = 'testing_parse.txt'
test_file = open(test_name)
outfile = open(test_parse,'w')
for i,line in enumerate(test_file):
	tmp = line.strip().split()
	remain = []
	head = ''
	for j in range(len(tmp)):
		if j == 0:
			continue
		if tmp[j][0] == '[':
			head = tmp[j][1:-1].lower()#.replace('-','')
		elif tmp[j]!='.' and tmp[j]!=',':
			remain.append(tmp[j].lower())#.replace('-',''))
	outfile.write(head)
	for x in remain:
		outfile.write(' '+x)
	outfile.write('\n')
outfile.write('EXIT\n')
outfile.close()
test_file.close()
