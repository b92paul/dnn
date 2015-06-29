import sys
f = open(sys.argv[1])
count = {}
for line in f:
  tmp = line.strip().split()
  for w in tmp:
    if w in count:
      count[w] += 1
    else:
      count[w] = 1
all_word = [(count[w],w) for w in count]
all_word.sort()
all_word.reverse()
for i in range(len(all_word)):
  print(all_word[i])
