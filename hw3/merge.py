import os
novel_path = 'Holmes_Training_Data/'
files = os.listdir(novel_path)
merge_novel = open('merge_novel.txt','w')
for i,filename in enumerate(files):
    if filename == '.gitignore':
        continue
    print i+1,filename
    novel = open(novel_path+filename)
    for line in novel:
        line = line.replace('-','')
        line = line.replace('\n','')
        line = line.lower()
        merge_novel.write(line)
    novel.close()
merge_novel.close()
