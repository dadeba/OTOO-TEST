require 'rake/clean'
################################################################################
$:.unshift File.expand_path(File.dirname(__FILE__))
################################################################################
LIB = '' if !defined?(LIB)
INC = '' if !defined?(INC)
################################################################################
require 'conf.rb'
################################################################################
OBJS  = (SRC).pathmap('%f').ext('o')
CLEAN.include([OBJS, "run", "build", "*.il", "*.isa", "*.file", "*.log", "*.cdf", "*.mod"])
CLOBBER.include("run")
################################################################################

target = {
  :DEFAULT => ['-O3', '-fopenmp'],
  :DEBUG   => ['-g',  ''],
  :PROFILE => ['-pg', ''],
  :CHECK   => ['-g -DCHECK',  ''],
}

$gen_makefile = false
$makefile_data = ""

task :default => :DEFAULT

target.each { |x,op|
  desc "build #{x}"
  task "#{x}", :cc, :sse do |t, args|

    options = {}
    options[:cc] = args[:cc] || "g++"
    options[:cflags] = " #{op[0]} #{op[1]} " + (args[:sse] || "-msse2")
    options[:lflags] = " #{op[1]} "

    case options[:cc]
    when "g++"
      options[:fc] = "gfortran"
    when "x86_64-unknown-linux-gnu-g++"
      options[:fc] = "x86_64-unknown-linux-gnu-gfortran"
    when "icpc"
      options[:fc] = "ifort"
#      options[:lflags] += " -nofor-main"  
    when /mpi/ 
      options[:fc] = options[:cc]
    else
      raise "no compiler #{options[:cc]}"
    end

    gen_task(options)
    Rake::Task["run"].invoke
  end
}

def gen_task(options)
  cflags = CFLAGS + ' '+ options[:cflags]
  lflags = LFLAGS + ' '+ options[:lflags]

  cc = options[:cc]
  fc = options[:fc]

  file "run" => OBJS do |t|
    sh "#{cc} #{t.prerequisites.join(' ')} #{lflags} -o #{t.name}"	
  end

  rule '.o' => ['.c'] do |t|
    sh "#{cc} #{cflags} -c #{t.prerequisites[0]} -o #{t.name}"
  end

  rule '.o' => ['.cpp'] do |t|
    sh "#{cc} #{cflags} -c #{t.prerequisites[0]} -o #{t.name}"
  end

  rule '.o' => ['.f'] do |t|
    sh "#{fc} -O3 -IWDMODEL -c #{t.prerequisites[0]} -o #{t.name}"
  end

  rule '.o' => ['.f90'] do |t|
    sh "#{fc} -O3 -c #{t.prerequisites[0]} -o #{t.name}"
  end

  rule '.hpp' do |t|
    if File.exist?(t.name) then
      touch t.name
    end
  end

  rule '.file' => ['.cl'] do |t|
    sh "./utils/template-converter #{t.name} #{t.prerequisites[0]} > #{t.name}"
  end

  current_options = options.to_s + cflags + lflags
  if check_options(current_options) then
    Rake::Task[:clean].invoke
    f = open("build", "w")
    f.write current_options
    f.close
   end
end

def check_options(options)
  str = ''
  begin
    f = open("build", "r")
    str = f.read
    f.close
    return !(options === str)	
  rescue
    return true
  end
end

def sh(args)
  if $gen_makefile
    $makefile_data << "\t#{args}\n"
  else
    super(args)
  end
end

#
# http://rfelix.com/2009/02/04/generate-makefile-with-rake/
# https://gist.github.com/57893
#
desc "Generates a Makefile to build the example code"
task :makefile do
  $gen_makefile = true
  $makefile_data << "all: main\n"
  
  $makefile_data << "main:\n"
  Rake::Task['default'].invoke
  
  $makefile_data << "clean:\n"
  CLEAN.each do |f|
    $makefile_data << "\trm -rf #{f}\n"
  end
  
  $makefile_data << "clobber:\n"
  CLEAN.each do |f|
    $makefile_data << "\trm -rf #{f}\n"
  end
  CLOBBER.each do |f|
    $makefile_data << "\trm -rf #{f}\n"
  end
  
  File.open('Makefile', 'w') { |f| f.write $makefile_data}
end
