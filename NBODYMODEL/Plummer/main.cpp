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

  class PlummerSphere : public InitialModel {
  public:
    PlummerSphere(uint64);
    ~PlummerSphere() {};
    void DumpModel();

  private:
    //    static const double pi = M_PI;
    static const uint64 buf_size = 1024;

    std::vector<double> rr, dd, mm;
    gsl_spline *r_spline;
    gsl_spline *d_spline;
  };

  PlummerSphere::PlummerSphere(uint64 n) : InitialModel(n) {
    rr.resize(buf_size);
    dd.resize(buf_size);
    mm.resize(buf_size);
  }

  void PlummerSphere::DumpModel()
  {
    double tm = 1.0;
    double pmas = tm/nall;
    double conv = 3.0*M_PI/16.0;
    std::cout << "pmas = " << pmas << "\n";

    uint64 i = 0;
    while(i < nall) {
      double X1 = my_rand();
      double X2 = my_rand();
      double X3 = my_rand();
      double R = 1.0/sqrt( (pow(X1,-2.0/3.0) - 1.0) );

      if(R < 100.0) {
	double Z = (1.0 - 2.0*X2)*R;
	double X = sqrt(R*R - Z*Z) * cos(2.0*M_PI*X3);
	double Y = sqrt(R*R - Z*Z) * sin(2.0*M_PI*X3);

	double Ve = sqrt(2.0)*pow( (1.0 + R*R), -0.25 );

	double X4 = 0.0;
	double X5 = 0.0;

	while( 0.1*X5 >= X4*X4*pow( (1.0-X4*X4), 3.5) ) {
	  X4 = my_rand(); X5 = my_rand();
	}

	double V = Ve*X4;
	double X6 = my_rand();
	double X7 = my_rand();

	double Vz = (1.0 - 2.0*X6)*V;
	double Vx = sqrt(V*V - Vz*Vz) * cos(2.0*M_PI*X7);
	double Vy = sqrt(V*V - Vz*Vz) * sin(2.0*M_PI*X7);

	X *= conv;
	Y *= conv;
	Z *= conv;    
	Vx /= sqrt(conv);
	Vy /= sqrt(conv);
	Vz /= sqrt(conv);

	ms[i] = pmas;

	px[i] = X;
	py[i] = Y;
	pz[i] = Z;
	
	vx[i] = Vx;
	vy[i] = Vy;
	vz[i] = Vz;

	i++;
	
	} /* if(r < 100.0) */
    } /* while(i < N) */

    std::stringstream new_file;
    new_file << "model_" << nall/1024 << ".cdf";
    WriteCDF(new_file.str().c_str());
  }
}

int main(int narg, char **argv)
{
  OTOO::PlummerSphere pl(atoi(argv[1]));
  pl.DumpModel();
}
