
class Mapper:
	def __init__(self):
		f = open('data/phones/state_48_39.map')
		self.map_48_39 = {}
		self.map_state_48 = {}
		self.map_state_39 = {}
		self.map_48 = {}
		self.list_48 = []
		self.map_39 = {}
		self.list_39 = []
		self.build_maps(f)

	def build_maps(self, file):
		count_48 = 0
		count_39 = 0
		for line in file:
			line = line.strip().split('\t')
			state = line[0]
			l48 = line[1]
			l39 = line[2]
			self.map_state_39[state] = l39
			self.map_state_48[state] = l48
			self.map_48_39[l48] = l39
			if line[1] not in self.map_48:
				self.map_48[line[1]] = count_48
				self.list_48.append(line[1])
				count_48 += 1
			if line[2] not in self.map_39:
				self.map_39[line[1]] = count_39
				self.list_39.append(line[1])
				count_39 += 1

	def ptrans(self, value, source = "48", target = "39"):
		if source == "48":
			if target == "39":
				return self.map_48_39[value]
		else:
			if target == "38":
				return self.map_state_38[value]
			else:
				return self.map_state_48[value]

	def get_index(self, value, type = "48"):
		if type == "48":
			return self.map_48[value]
		else:
			return self.map_39[value]

	def get_phone(self, index, type = "48"):
		if type == "48":
			return self.list_48[index]
		else:
			return self.list_39[index]

def main():
	mapper = Mapper()
	print mapper.trans('ao')
	print mapper.trans('1024', 'state', '39')

if __name__ == '__main__':
	main()
