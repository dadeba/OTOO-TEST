#ifndef OTOO_INTEGRATE_H
#define OTOO_INTEGRATE_H
#include "OTOO.hpp"

namespace OTOO {
  template<typename AR>
  void Swap(AR &src, AR &dst, const uint64 *l)
  {
    for(uint64 i = 0; i < (uint64)src.rows(); i++) {
      dst.row(i) = src.row(l[i]);
    }
    src = dst;
  }

  template<typename AR>
  void Swap1(AR &src, AR &dst, const uint64 *l)
  {
    for(uint64 i = 0; i < (uint64)src.rows(); i++) {
      dst(i) = src(l[i]);
    }
    src = dst;
  }

  template<typename AR, typename AR2>
  void force(uint64 nall, AR &x, AR2 &m, float e2, AR &a, AR2 &p)
  {
    for(uint64 i = 0; i < nall; i++) {
      //      Vector3 acc;
      Eigen::Vector3d acc;
      double pt = 0.0;
      acc << 0.0, 0.0, 0.0;
      //      p(i) = 0.0;
      for(uint64 j = 0; j < nall; j++) {
	Vector3 dx, ax;
	float r1i, r3i, mj, r2; 

	if (i == j) continue;

	mj = m(j);

	dx = x.row(j) - x.row(i);
	r2 = dx.dot(dx);

	r1i = 1.0/sqrt(r2+e2);
	r3i = r1i*r1i*r1i;
	ax  = mj*dx*r3i;
	acc.x() += ax.x(); 
	acc.y() += ax.y(); 
	acc.z() += ax.z(); 
	pt -= mj*r1i;
	//	acc += mj*dx*r3i;
	//	p(i) -= mj*r1i;

      }
      //      a.row(i) = acc;
      a.row(i).x() = acc.x();
      a.row(i).y() = acc.y();
      a.row(i).z() = acc.z();
      p(i)     = pt;
    }
  }

  template<typename AR, typename AR2>
  void E(AR &x, AR2 &m, AR &v, AR &a, AR2 &p, float &ke, float &pe, uint64 nall)
  {
    ke = 0.0;
    double tmp = 0.0;
    for(uint64 i = 0; i < nall; i++) {
      Vector3 v0 = v.row(i);
      tmp += m(i)*v0.squaredNorm();
    }
    ke = tmp;
    ke /= 2.0;
    pe = (m*p).sum()/2.0;
  }

  template<typename AR, typename AR2>
  void Check(AR &a0, AR2 &p0, AR &a1, AR2 &p1)
  {
    Vector3 a_err = (std::abs((a0 - a1)/a0)).colwise().sum();
    float   p_err = (std::abs((p0 - p1)/p0)).sum();
    std::cout << "Check " << (a_err/a0.rows()).transpose() << " " << p_err/p0.rows() << "\n";
  }

  class Integrate {
  public:
    Integrate();
    ~Integrate() {};

    //    virtual void Kick()  = 0;
    //    virtual void Drift() = 0;
    //    virtual void Setup() = 0;
    //    virtual void Loop()  = 0;

    virtual void AllocateVariables(uint64 n) {
      Allocate(n, x);
      Allocate(n, v);
      Allocate(n, a);
      Allocate(n, m);
      Allocate(n, p);

      Allocate(n, tmpX3);
      Allocate(n, tmpX);
      i_sort = new uint64[n];
    }

    void Read(const char *filename) {
      FILE *inp = fopen(filename, "r");

      if (inp == NULL) {
	std::cerr << "no input file : " << filename << "\n";
	exit(-1);
      }

      fscanf(inp, "%lld\n", &nall);
      double dum;
      fscanf(inp, "%lE\n", &dum);

      AllocateVariables(nall);

      // read initial data
      for(uint64 i = 0; i < nall; i++) {
	fscanf(inp,"%f %f %f %f %f %f %f\n", 
	       &m(i), &x(i,0), &x(i,1), &x(i,2), &v(i,0), &v(i,1), &v(i,2));
      }
      fclose(inp);
    };

    void SetTend(const float tt) {
      std::cout << "# set Tend = " << tt << "\n";
      tend = tt;
    };

    void SetdT(const float tt) {
      std::cout << "# set dt = " << tt << "\n";
      dt = tt;
    };

    void SetdTsys(const float tt) {
      std::cout << "# set dt_sys = " << tt << "\n";
      dt_sys = tt;
    };

    void Out(float te1, float ke1, float pe1) {
      std::cerr << t << "\t" << dt << "\t" << te1 << "\t" << ke1 << "\t" << pe1 << "\t" << (te0-te1)/te0 << "\n";
    };

    void SwapX3(ArrayX3 &src)
    {
      OTOO::Swap(src, tmpX3, i_sort); 
    }

    void SwapX(ArrayX  &src)
    {
      OTOO::Swap1(src, tmpX, i_sort); 
    }

    uint64 GetNumberOfParticles() {
      return nall;
    }

    uint64 GetNSTEP() {
      return nstep;
    }; 
    
    double GetNQ() {
      return nstep*(double)nall*(double)nall;
    }; 

    void SetUnit(double, double, double);
    void SetInitialGravityTolerance();
    void CalcGravityTolerance(ArrayX3 &, float);

    uint64 nall;
    ArrayX3 x, v, a;
    ArrayX m, p;
    float t, tend, dt, dt_sys;
    float dt_dump;
    uint64 nstep; 
    float ke, pe, te;
    float ke0, pe0, te0;
    // temporary buffers
    //    ArrayX4 tmpX4;
    ArrayX3 tmpX3;
    ArrayX  tmpX;
    // for sort index
    uint64 *i_sort;

    double start_time_loop;
    double elapsed_time_loop;
    std::ofstream log_energy;

    // parameters
    float grav_err_max;
    double grav;
    double lunit, munit, tunit;
    double vunit, eunit, dunit;
    double punit;
    //    double divunit;
    double aunit, deunit;
    double lunit_inv;
    double munit_inv;
    double vunit_inv;
    double dunit_inv;
    double punit_inv;
    double eunit_inv;
    float eps;
    float err_max;
    float m_frac;
    //    uint64 grav_VL;

    /*
      Physical constant in cgs unit, from Galactic Dynamics 643p 

      bk : bolzman's constant
      pc : parsec in cm
      sm : solar mass in g
      sr : solar radius in cm
      yr : one year in sec
      gc : gravitational constant
      pm : proton mass
      em : electron mass
      pi : 3.14159265
    */
    static const double bk;
    static const double pc;
    static const double sm;
    static const double sr;
    static const double yr;
    static const double gc;
    static const double pm;
    static const double em;
    static const double pi;
  };
  const double Integrate::bk = 1.38066e-16;
  const double Integrate::pc = 3.0857e18;
  const double Integrate::sm = 1.9891e33;
  const double Integrate::sr = 6.9599e10;
  const double Integrate::yr = 3.1536e7;
  const double Integrate::gc = 6.672e-8;
  const double Integrate::pm = 1.672649e-24;
  const double Integrate::em = 9.10953e-28;
  const double Integrate::pi = 3.14159265;

  Integrate::Integrate()
  {
    std::cout.precision( 6 );
    std::cout.setf( std::ios_base::scientific, std::ios_base::floatfield );
    std::cerr.precision( 4 );
    std::cerr.setf( std::ios_base::scientific, std::ios_base::floatfield );

    log_energy.open("energy.log");
    log_energy.precision( 6 );
    log_energy.setf( std::ios_base::scientific, std::ios_base::floatfield );
  }

  void Integrate::SetUnit(double lu, double mu, double tu)
  {
    lunit = lu;
    munit = mu;
    tunit = tu;

    vunit = lunit/tunit;
    dunit = munit/pow(lunit, 3.0);

    eunit = vunit*vunit;      // ----> erg/g
    punit = munit*lunit/pow(tunit, 2.0)/pow(lunit, 2.0);

    //    divunit = dunit*(vunit/lunit); // dn * v/l
    aunit  = vunit/tunit;             
    deunit = eunit/tunit;

    std::cout << "# dunit " << dunit << "\n"; 
    std::cout << "# eunit " << eunit << "\n"; 
    std::cout << "# punit " << punit << "\n"; 

    lunit_inv = 1.0/lunit;
    munit_inv = 1.0/munit;
    vunit_inv = 1.0/vunit;
    dunit_inv = 1.0/dunit;
    eunit_inv = 1.0/eunit;
    punit_inv = 1.0/punit;
  }

  void Integrate::SetInitialGravityTolerance()
  {
    Vector3 c1;
    float total_mass = m.sum();
    float m_t = total_mass/nall;
    
    Vector3 pmax = x.colwise().maxCoeff();
    Vector3 pmin = x.colwise().minCoeff();
    Vector3 box  = pmax - pmin;
    //    float   r_l = box.maxCoeff()/1.0e4;
    float   r_l = box.maxCoeff()/3.0;

    grav_err_max = m_t/(r_l*r_l);
    grav_err_max = sqrt(total_mass);

    //    std::cout << "# Total Mass   " << total_mass << "\n";
    //    std::cout << "# Length scale " << r_l << "\n";
    std::cout << "# initial g tolerance  " << grav_err_max << "\n";
  }

  void Integrate::CalcGravityTolerance(ArrayX3 &a_j, float f)
  {
    float a_mean = 0.0;
    for(uint64 j = 0; j < nall; j++) {
      Vector3 aj = a_j.row(j);
      a_mean += aj.norm();
    }
    a_mean /= f;
    a_mean /= nall;
    grav_err_max = a_mean*err_max;
  }
}
#endif
