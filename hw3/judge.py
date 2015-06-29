import sys
if len(sys.argv) != 3:
	sys.exit('input two filename!')
filename1 = sys.argv[1]
filename2 = sys.argv[2]
f1 = open(filename1)
f2 = open(filename2)
count = 0
num = 0.0
x1 = []
x2 = []
for line in f1:
	x1.append(line.strip())
for line in f2:
	x2.append(line.strip())
for x,y in zip(x1,x2):
	count = count +1
	if x == y:
		num = num + 1
print "%lf %d,%lf\n" % (num,count-1,num/(count-1))
