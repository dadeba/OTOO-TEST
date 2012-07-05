#ifndef OTOO_NBODYMODEL_H
#define OTOO_NBODYMODEL_H
#include <netcdfcpp.h>

#include "Integrate.hpp"
#include "OcTreeOpenCL.hpp"
#include "Config.hpp"

namespace OTOO {
#include "kernel_nbody.file"
  class NBODYmodel : public Integrate {
public:
    NBODYmodel(const char *);
    ~NBODYmodel();
    void Setup(uint64, uint64);
    void Loop();
    double GetTotalTime();
    double GetKernelTime1();
    double GetKernelTime2();

  private:
    void AllocateVariables(uint64);
    void CalcDT();
    void Sort();
    void Force();
    void Out();
    void Eall();
    void ReadCDF(const char *);
    void WriteCDF(const char *);
    uint64 ReadNfromCDF(const char *);
    float dt_nbody; 
    GravityTree *T;
    ArrayX3 v_old;

    uint64 count_force;
    double accum_time_ct, accum_time_cm, accum_time_b2, accum_time_gk;
    double accum_time_force, accum_time_total;

    void ClearCounters();

    ConfigFile CF;
  };

  NBODYmodel::~NBODYmodel()
  {
  }

  NBODYmodel::NBODYmodel(const char *filename)  : Integrate::Integrate()
  {
    // parameters
    CF.Load("NBODY.conf");
    lunit = munit = tunit = 1.0; 
    eps     = CF.fGet("eps");
    err_max = grav_err_max = CF.fGet("err_max");
    m_frac  = CF.fGet("m_frac");
    dt_sys    = CF.fGet("dt_sys");
    dt_dump   = CF.fGet("dt_dump");
    dt_nbody  = CF.fGet("dt_nbody");

    SetTend( CF.fGet("tt_end") );

    SetUnit(lunit, munit, tunit);
    grav = 1.0/(pow(lunit, 3.0)*pow(tunit, 2.0)*pow(munit,-1.0));

    nall = ReadNfromCDF(filename);
    AllocateVariables(nall);
    T = new GravityTree(nall); 
    T->SetEPS(eps*lunit_inv);

    ReadCDF(filename);
  }

  void NBODYmodel::AllocateVariables(uint64 n)
  {
    Integrate::AllocateVariables(n);
    OTOO::Allocate(n, v_old);
  }

  void NBODYmodel::Setup(uint64 pl = 0, uint64 d = 0)
  {
    T->SetVL( CF.fGet("grav_VL") );
    T->SetupOpenCL(pl, d, kernel_nbody_file);
    T->ConstructKeybasedTree(nall, x, m);
    std::cerr << "\n";

    Sort();

    SetInitialGravityTolerance();
    Force();
    CalcGravityTolerance(a, grav);
    Force();

    Eall();
    te0 = te;
    Out();

    T->ClearTimingInfo();
    ClearCounters();
  }

  void NBODYmodel::CalcDT()
  {
    dt = dt_nbody;
  }

  void NBODYmodel::Force() {
    double t1 = e_time();
    T->ConstructKeybasedTree(nall, x, m);
    double t2 = e_time();
    T->CalcCM();
    double t3 = e_time();
    T->SetALLEPS2();
    T->CalcB2(grav_err_max, m_frac);
    double t4 = e_time();
    T->CalcGravityOpenCL(a, p);
    a *= grav;
    p *= grav;
    double t5 = e_time();

    std::cerr << "\n";

    count_force++;
    accum_time_ct += t2-t1;
    accum_time_cm += t3-t2;
    accum_time_b2 += t4-t3;
    accum_time_gk += t5-t4;
    accum_time_force += t5 - t1;
  }

  void NBODYmodel::ClearCounters()
  {
    count_force = 0;
    accum_time_force = 0.0;
    accum_time_ct = accum_time_cm = accum_time_b2 = accum_time_gk = 0.0;
    accum_time_total = 0.0;
  }

  double NBODYmodel::GetTotalTime()
  {
    return accum_time_total/count_force;
  }

  double NBODYmodel::GetKernelTime1()
  {
    return accum_time_gk/count_force;
  }

  double NBODYmodel::GetKernelTime2()
  {
    return T->GetKernelTime(0);
  }

  void NBODYmodel::Sort()
  {
    T->GetMortonOrder(i_sort);
    SwapX3(x);
    SwapX3(v);
    SwapX3(a);
    SwapX(m);
  }

  void NBODYmodel::Eall()
  {
    E(x, m, v, a, p, ke, pe, nall);
    te = ke + pe;
  }

  void NBODYmodel::Loop()
  {
    start_time_loop = e_time();
    while(t <= tend) {
      double t1 = e_time();
      CalcDT();

      std::cerr << t << " " << dt << " ";

      x += v*dt + 0.5f*a*dt*dt; 
      v_old  = v + 0.5f*a*dt;

      // predict 
      v += a*dt;

      Force();

      // correct
      v = v_old + 0.5f*a*dt;
      
      accum_time_total += e_time()  - t1;

      t += dt;
      nstep++;
      if (fmod(t, dt_sys) == 0.0) {
	CalcGravityTolerance(a, grav);
	Sort();
	Eall();
	Out();
      }
      if (fmod(t, dt_dump) == 0.0) {
	char buf[100];
	sprintf(buf, "%i.cdf", (int)(t/dt_dump));
	WriteCDF(buf);
      }
    }

    Eall();
    Out();
  }

  void NBODYmodel::Out()
  {
    std::cout << "# " << nstep << " " << t << "\t" << dt << "\t" << te << "\t" << ke << "\t" << pe << "\t"
	      << 100.0f*(te0-te)/te0 << " %\n";
    std::cout << "# g tolerance  " << grav_err_max << " (" << err_max*100.0 << " %)\n";

    elapsed_time_loop = e_time() - start_time_loop;

    std::cerr << "CT "    << accum_time_ct/count_force
	      << "\tCM " << accum_time_cm/count_force
	      << "\tB2 " << accum_time_b2/count_force
	      << "\tGK " << accum_time_gk/count_force
	      << "\tFF " << accum_time_force/count_force
	      << "\tTOTAL " << accum_time_total/count_force
	      << "\n";

    T->DumpTimingInfo();

    log_energy
      << nstep << " " 
      << t*tunit << " " 
      << dt*tunit << " " 
      << (double)te*eunit << " "
      << (double)ke*eunit*munit << " " 
      << (double)pe*eunit*munit << " " 
      << (elapsed_time_loop)/3600.0 << "\n";
    log_energy.flush();
  }

  uint64 NBODYmodel::ReadNfromCDF(const char *filename) 
  {
    NcFile f(filename, NcFile::ReadOnly);
    if (!f.is_valid()) {
      std::cerr << "ReadCDF failed\n";
      exit(-1);
    }
    int i;
    f.get_var("nall")->get(&i, 1);
    return (uint64)i;
  }

  void NBODYmodel::ReadCDF(const char *filename) 
  {
    NcFile f(filename, NcFile::ReadOnly);
    if (!f.is_valid()) {
      std::cerr << "ReadCDF failed\n";
      return;
    }

    int n;
    f.get_var("nstep")->get(&n, 1);;
    nstep = n;
    f.get_var("time")->get(&t, 1);;
    std::cout << "# Read CDF file at time = " << t << "\n";

    double *buf_x = new double[nall];
    double *buf_y = new double[nall];
    double *buf_z = new double[nall];

    f.get_var("px")->get(buf_x, nall);
    f.get_var("py")->get(buf_y, nall);
    f.get_var("pz")->get(buf_z, nall);
    for(uint64 i = 0; i < nall; i++) {
      x(i,0) = buf_x[i]*lunit_inv;
      x(i,1) = buf_y[i]*lunit_inv;
      x(i,2) = buf_z[i]*lunit_inv;
    }

    f.get_var("vx")->get(buf_x, nall);
    f.get_var("vy")->get(buf_y, nall);
    f.get_var("vz")->get(buf_z, nall);
    for(uint64 i = 0; i < nall; i++) {
      v(i,0) = buf_x[i]*vunit_inv;
      v(i,1) = buf_y[i]*vunit_inv;
      v(i,2) = buf_z[i]*vunit_inv;
    }

    f.get_var("ms")->get(buf_x, nall);
    for(uint64 i = 0; i < nall; i++) {
      m(i)   = buf_x[i]*munit_inv;
    }

    delete buf_x;
    delete buf_y;
    delete buf_z;
  }

  void NBODYmodel::WriteCDF(const char *filename) 
  {
    NcFile out(filename, NcFile::Replace);
    if (!out.is_valid()) {
      std::cerr << "WriteCDF failed\n";
      return;
    }

    int n = nstep;
    out.add_var("nstep", ncInt)->put(&n, 1);;
    n = nall;
    out.add_var("nall", ncInt)->put(&n, 1);
    out.add_var("time", ncDouble)->put(&t, 1);;
    out.add_var("dt_sys", ncDouble)->put(&dt_sys, 1);;

    NcDim *na = out.add_dim("na", nall);
    double *buf_x = new double[nall];
    double *buf_y = new double[nall];
    double *buf_z = new double[nall];

    for(uint64 i = 0; i < nall; i++) {
      buf_x[i] = x(i,0)*lunit;
      buf_y[i] = x(i,1)*lunit;
      buf_z[i] = x(i,2)*lunit;
    }
    out.add_var("px", ncDouble, na)->put(buf_x, nall);
    out.add_var("py", ncDouble, na)->put(buf_y, nall);
    out.add_var("pz", ncDouble, na)->put(buf_z, nall);

    for(uint64 i = 0; i < nall; i++) {
      buf_x[i] = v(i,0)*vunit;
      buf_y[i] = v(i,1)*vunit;
      buf_z[i] = v(i,2)*vunit;
    }
    out.add_var("vx", ncDouble, na)->put(buf_x, nall);
    out.add_var("vy", ncDouble, na)->put(buf_y, nall);
    out.add_var("vz", ncDouble, na)->put(buf_z, nall);

    for(uint64 i = 0; i < nall; i++) {
      buf_x[i] = m(i) *munit;

    }
    out.add_var("ms", ncDouble, na)->put(buf_x, nall);

    delete buf_x;
    delete buf_y;
    delete buf_z;
  }
}
#endif
