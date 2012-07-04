#include <netcdfcpp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_spline.h>
#include "OTOO.hpp"

namespace OTOO {
  class InitialModel {
  public:
    InitialModel(uint64 n) : nall(n*1024) {
      ranran_setup(2012);
      px.resize(nall);
      py.resize(nall);
      pz.resize(nall);
      vx.resize(nall);
      vy.resize(nall);
      vz.resize(nall);
      ms.resize(nall);
    };
    ~InitialModel() {};

  protected:
    uint64 nall;
    std::vector<double> px, py, pz, vx, vy, vz, ms;
    void WriteCDF(const char *filename)
    {
      NcFile out(filename, NcFile::Replace);
      if (!out.is_valid()) {
	std::cerr << "WriteCDF failed\n";
	return;
      }

      std::cout << "Dumping nall = " << nall << "\n";

      int n = 0;
      out.add_var("nstep", ncInt)->put(&n, 1);;
      n = nall;
      out.add_var("nall", ncInt)->put(&n, 1);
      double t = 0.0;
      out.add_var("time", ncDouble)->put(&t, 1);;

      NcDim *na = out.add_dim("na", nall);
      out.add_var("px", ncDouble, na)->put(&px[0], nall);
      out.add_var("py", ncDouble, na)->put(&py[0], nall);
      out.add_var("pz", ncDouble, na)->put(&pz[0], nall);
      out.add_var("vx", ncDouble, na)->put(&vx[0], nall);
      out.add_var("vy", ncDouble, na)->put(&vy[0], nall);
      out.add_var("vz", ncDouble, na)->put(&vz[0], nall);
      out.add_var("ms", ncDouble, na)->put(&ms[0], nall);
    }

    const gsl_rng_type *T;
    gsl_rng * r;
    void ranran_setup(int s)
    {
      gsl_rng_env_setup();
      T = gsl_rng_default;
      r = gsl_rng_alloc (T);
      std::cerr << gsl_rng_name (r) << "\n";
      gsl_rng_set(r, s);
    }
    double d_rand(void)
    {
      return gsl_rng_uniform_pos(r);
    }
    double d_2rand(void)
    {
      return 2.0*(d_rand()-0.5);
    }
    double my_rand()
    {
      return d_rand();
    }
  };

  class UniformSphere : public InitialModel {
  public:
    UniformSphere(uint64);
    ~UniformSphere() {};
    void DumpModel();

  private:
    //    static const double pi = M_PI;
    static const uint64 buf_size = 1024;

    std::vector<double> rr, dd, mm;
    gsl_spline *r_spline;
    gsl_spline *d_spline;
  };

  UniformSphere::UniformSphere(uint64 n) : InitialModel(n) {
    rr.resize(buf_size);
    dd.resize(buf_size);
    mm.resize(buf_size);
  }

  void UniformSphere::DumpModel()
  {
    double tm = 1.0;
    double r0 = 1.0;
    double pmas = tm/nall;
    std::cout << "pmas = " << pmas << "\n";

    uint64 i = 0;
    while(i < nall) {
      double a = d_2rand();
      double b = d_2rand();
      double c = d_2rand();
      double d = a*a+b*b+c*c;
      if (d <= 1.0) {
	ms[i] = pmas;

	px[i] = a*r0;
	py[i] = b*r0;
	pz[i] = c*r0;
	
	vx[i] = 0.0;
	vy[i] = 0.0;
	vz[i] = 0.0;
	i++;
      }
    }

    std::stringstream new_file;
    new_file << "model_" << nall/1024 << ".cdf";
    WriteCDF(new_file.str().c_str());
  }
}

int main(int narg, char **argv)
{
  OTOO::UniformSphere pl(atoi(argv[1]));
  pl.DumpModel();
}
