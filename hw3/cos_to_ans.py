f = open('cos.txt')
ans = open('ans.csv','w')
ans.write('id,answer\n')
eng = 'abcde'
best = -2.0
idx = -1
for i,line in enumerate(f):
	now = float(line)
	if now > best:
		best,idx = now,i%5
	if (i) % 5 == 4:
		ans.write('%d,%c\n' % (i/5+1, eng[idx]))
		best,idx = -2.0,-1
ans.close()
f.close()
