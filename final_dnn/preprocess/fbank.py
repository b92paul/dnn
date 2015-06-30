import math
data_path = '../../data/final/'
check = 100000
train_fbank_path = data_path + 'train.ark'
train_label_path = data_path + 'train.lab'

train_fbank_file = open(train_fbank_path)
train_label_file = open(train_label_path)

norm_file = open('preprocess/norm.out')

bias = []
var = []
for line in norm_file:
  tmp = line.strip().split()
  print(tmp)
  b, v = float(tmp[0]), float(tmp[1])
  bias.append(b)
  var.append(math.sqrt(v**2 - b**2))

def preprocess_data(limit=-1):
  train_fbank_out = open(data_path + 'f_train_n.ark', 'w')
  train_label_out = open(data_path + 'f_1943.lab', 'w')
  labels = {}
  for i,line in enumerate(train_label_file):
    tmp = line.strip().split(',')
    labels[tmp[0]] = tmp[1]
  for line in train_fbank_file:
    tmp = line.strip().split()
    for i in range(1,len(tmp)):
      res = (float(tmp[i]) - bias[i-1])/var[i-1]
      train_fbank_out.write('%.6f ' % res)
    train_fbank_out.write('\n')
    train_label_out.write(labels[tmp[0]]+'\n')
def preprocess_test():
  test_fbank_file = open(data_path + 'test.ark')
  test_fbank_out = open(data_path + 'f_test_n.ark', 'w')
  test_id_out = open(data_path + 'test_id.ark', 'w')
  for i,line in enumerate(test_fbank_file):
    tmp = line.strip().split()
    test_id_out.write(tmp[0]+'\n')
    for i in range(1,len(tmp)):
      res = (float(tmp[i]) - bias[i-1])/var[i-1]
      test_fbank_out.write('%.6f ' % res)
    test_fbank_out.write('\n')
    
if __name__ == '__main__':
  preprocess_data()
  preprocess_test()

