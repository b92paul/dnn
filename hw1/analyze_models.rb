#!/usr/bin/ruby


$models = []

def file_to_obj(filename)
  ret = {}
  ret[:model] = {}  
  ret[:model][:neuron]=filename.scan(/[0-9]+_[0-9]+_[0-9]+_[0-9]+/)[0].split('_')[0..-2].map(&:to_i)
  ret[:model][:eta] = filename.scan(/[0-9]+_[0-9]+_[0-9]+_[0-9]+/)[0].split('_')[-1].to_f / 10.0
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
  "#{e[:model][:neuron].join(',')} #{e[:model][:eta]}: #{e[:data][:max]}"
end

def analyze
  puts "Best rate of all: #{print_model($models.max_by{|c| c[:data][:max]})}"
  puts "Best eta of each:"
  $models.group_by{|c|c[:model][:neuron]}.each{|model,models|
    puts "  Best eta of #{model}: #{models.max_by{|c|c[:data][:max]}[:model][:eta]}"
  }
end

def main
  Dir.chdir('models')
  Dir.glob("*.res").each{|filename|
    $models << file_to_obj(filename)
  }
  analyze
end

main

