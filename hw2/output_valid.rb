#!/usr/bin/ruby
$path = "../../data/"
$valid_ark = $path + 'valid_0.ark'

def read_example(data)
	obj={}
  frame,len = data[0].split(' ').map(&:to_i)
	fail if len != 69
	obj[:label] = data[1]
	obj[:x] = []
	for i in 2..frame+1
		obj[:x] << data[i]
  end
	obj[:y] = data[frame+2]

	obj[:frame] = frame
	obj[:len] = len
	return obj
end
def main(filename)
  data = IO.read(filename).split("\n")
	n = data[0].to_i
	data.shift
	puts "n = %d" % n
	examples = []
	for i in 1..n
    examples << read_example(data)
	  data.shift(examples[-1][:frame]+3)
	end
	fail if examples.size != n
	File.open('valid_0.output','w'){|f|
	  for ex in examples
			f.puts ex[:y]
		end
	}
	File.open('valid_0.label','w'){|f|
	  for ex in examples
		  f.puts ex[:label]
		end
	}
	print "Generating validation output\n"
	print %x(cd DataUtils; python output_processor.py .. valid_0)
	print %x(rm valid_0.label valid_0.output)
end

main($valid_ark)
