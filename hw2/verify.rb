#!/usr/bin/ruby

def edit_distance(s,t)
  m = s.length
  n = t.length
  return m if n == 0
  return n if m == 0
  d = Array.new(m+1) {Array.new(n+1)}

  (0..m).each {|i| d[i][0] = i}
  (0..n).each {|j| d[0][j] = j}
  (1..n).each do |j|
    (1..m).each do |i|
      d[i][j] = if s[i-1] == t[j-1]  # adjust index into string
                  d[i-1][j-1]       # no operation required
                else
                  [ d[i-1][j]+1,    # deletion
                    d[i][j-1]+1,    # insertion
                    d[i-1][j-1]+1,  # substitution
                  ].min
                end
    end
  end
  d[m][n]
end

def calc(t1,t2)
  l1,s1 = t1.split(',')
	l2,s2 = t2.split(',')
	if l1 != l2
		puts "Label not the same! QQ? #{l1} <> #{l2}"
	  exit
	end
	edit_distance(s1,s2)
end

def read(file)
  IO.read(file).split("\n")[1..-1]
end

def main(file1, file2='./valid_correct.csv')
  obj1 = read(file1)
	obj2 = read(file2)
	print "n1 = #{obj1.size}, n2 = #{obj2.size}\n"
	if obj1.size != obj2.size
	  puts "Error! two file has different n"
		exit
	end
	obj1.sort!
	obj2.sort!
	total = obj1.zip(obj2).inject(0){|sum,v|sum+calc(v[0],v[1])}
	print "Average edit distance = #{total/obj1.size.to_f}\n"
end

if ARGV.size == 1
	main(ARGV[0])
elsif ARGV.size == 2
  main(ARGV[0],ARGV[1])
else
	puts "USAGE: ./verify.rb file1 [file2=./valid_correct.csv]"
end
