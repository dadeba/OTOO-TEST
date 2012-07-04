#################################################################################
SRC = FileList["main_nbody.cpp"]

file "main_nbody.o" => ["main_nbody.cpp", "NBODYmodel.hpp", "kernel_nbody.file",
                        "OTOO/Integrate.hpp", 
                        "OTOO/OcTreeOpenCL.hpp",
                        "OTOO/Config.hpp"]
#################################################################################
INC = '-IOTOO'
#################################################################################
require 'common.rb'
