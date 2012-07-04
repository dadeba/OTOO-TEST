#include "NBODYmodel.hpp"

using namespace OTOO;

int main(int narg, char **argv)
{
  if (narg != 4) return -1;

  NBODYmodel NB(argv[1]);
  NB.Setup(atoi(argv[2]), atoi(argv[3]));

  double dum = e_time();
  NB.Loop();
  dum = e_time() - dum;
  std::cout << "# N \t steps \t time \t time/step \t total \t kernel \t kernel2 \n";
  std::cout << NB.GetNumberOfParticles() << "\t" 
	    << NB.GetNSTEP() << "\t"
	    << dum << "\t"
	    << dum/NB.GetNSTEP() << "\t"
	    << NB.GetTotalTime() << "\t"
	    << NB.GetKernelTime1() << "\t"
	    << NB.GetKernelTime2() << "\n";
}
