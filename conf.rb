#################################################################################
EIGEN  = "/opt/include"
OCLSDK = "/opt/AMDAPP" 
CFLAGS = "-std=c++0x -Wall -Wno-unused-result -I#{EIGEN} -I#{OCLSDK}/include -IOTOO #{INC}"
LFLAGS = "-lnetcdf -lnetcdf_c++ -lgsl -lgslcblas -lm -lOpenCL -L#{OCLSDK}/lib/x86_64"
#################################################################################
