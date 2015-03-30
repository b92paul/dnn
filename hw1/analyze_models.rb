#!/usr/bin/ruby

$dir = 'models'
$models = []

def file_to_obj(filename)
  ret = {}
  ret[:model] = {} 
  char = filename.scan(/[0-9][[0-9]*_]+[0-9]+.res$/)[0].split('_')
  ret[:model][:neuron]=filename.scan(/\[.*\]/)[0][1..-2].split('_').map(&:to_i)
  ret[:model][:eta] = char[-2].to_f / 10.0
  ret[:model][:momentom] = char[-1].to_f / 10.0
  ret[:data] = {}
  ret[:data][:raw] = []
  File.open(filename,'r'){|f|
    while (line=f.gets) != nil
      e_val = line.split(' ').first.to_f
      ret[:data][:raw]  << e_val
    end
  }
  ret[:data][:max] = ret[:data][:raw].max
  ret
end

def print_model(e)
  "#{e[:model][:neuron].join(',')} #{e[:model][:eta]} #{e[:model][:momentom]}: #{e[:data][:max]}"
end

def simple_statistics
  puts "Best rate of all: #{print_model($models.max_by{|c| c[:data][:max]})}"
  puts "Best eta of each:"
  $models.sort_by!{|c|-c[:data][:max]}
  $models.group_by{|c|c[:model][:neuron]}.each{|model,models|
    max = models.max_by{|c|c[:data][:max]}
    puts "  Best eta of #{model}: eta=#{max[:model][:eta]} mom=#{max[:model][:momentom]}"
  }
end

def output_csv
#  puts '"neuron","eta",' + (1..20).to_a.map{|c|c*1000}.join(',')
  $models.sort_by{|c|c[:model][:neuron].to_s+c[:model][:eta].to_s}.each{|f|
    puts "\"#{f[:model][:neuron]}\",\"#{f[:model][:eta]}\"," + f[:data][:raw].join(',')
  }
end


def help
  puts <<HELP
=====================================================
usage: ./analyze_models.rb command

    avalid commands are:

        s : output simple statistics
      csv : output e_val in csv format of each model

=====================================================
HELP
  exit
end

def load_files
  Dir.chdir($dir)
  Dir.glob("*.res").each{|filename|
    $models << file_to_obj(filename)
  }
end


def main
  help if ARGV.size != 1
  load_files
  case ARGV[0]
    when 's';simple_statistics
    when 'csv';output_csv
  end
end

main
