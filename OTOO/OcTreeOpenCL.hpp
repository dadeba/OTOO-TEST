#ifndef OTOO_GRAVITYTREE_H
#define OTOO_GRAVITYTREE_H
#include "OpenCLDevice.hpp"

namespace OTOO {
  struct PHKeys {
    static const uint64 T0[12][8];
    static const uint64 T1[12][8];

    uint64 poi(uint64 key, uint64 level)
    {
      return (key >> 3*level)&0x7ULL;
    }

    uint64 Get(uint64 morton_key, uint64 maxlevel)
    {
      uint64 key = 0;
      uint64 l = 0;
      for(uint64 level = maxlevel; level >= 1; level--) {
	//      for(uint64 level = maxlevel; level >= 5; level--) {
	uint64 p = poi(morton_key, level);
	uint64 oo;
	if (level >= maxlevel-1) {
	  oo = p;
	} else {
	  oo = T0[l][p];
	}
	key |= (oo << (level*3));
	l = T1[l][p];
      }
      return key;
    }
  };
  // A B C D E F G H I J K  M
  // 0 1 2 3 4 5 6 7 8 9 10 11
  const uint64 PHKeys::T0[12][8] = {
    {0, 2, 3, 1, 5, 7, 6, 4}, 
    {0, 1, 5, 4, 6, 7, 3, 2}, 
    {0, 4, 6, 2, 3, 7, 5, 1}, 
    {6, 4, 5, 7, 3, 1, 0, 2},
    {6, 7, 3, 2, 0, 1, 5, 4},
    {6, 2, 0, 4, 5, 1, 3, 7},
    {5, 7, 6, 4, 0, 2, 3, 1},
    {5, 4, 0, 1, 3, 2, 6, 7},
    {5, 1, 3, 7, 6, 2, 0, 4},
    {3, 1, 0, 2, 6, 4, 5, 7},
    {3, 2, 6, 7, 5, 4, 0, 1},
    {3, 7, 5, 1, 0, 4, 6, 2}
  };
  const uint64 PHKeys::T1[12][8] = {
    {1, 2, 2, 9, 9, 8, 8, 4},
    {2, 0, 0, 7, 7, 3, 3, 11},
    {0, 1, 1, 5, 5, 10, 10, 6},
    {4, 5, 5, 6, 6, 11, 11, 1},
    {5, 3, 3, 10, 10, 0, 0, 8}, 
    {3, 4, 4, 2, 2, 7, 7, 9},
    {7, 8, 8, 3, 3, 2, 2, 10},
    {8, 6, 6, 1, 1, 9, 9, 5},
    {6, 7, 7, 11, 11, 4, 4, 0},
    {10, 11, 11, 0, 0, 5, 5, 7},
    {11, 9, 9, 4, 4, 6, 6, 2},
    {9, 10, 10, 8, 8, 1, 1, 3}
  };

  class GravityTree {
  public:
    GravityTree(uint64);
    ~GravityTree();
    void SetupOpenCL(int, int, const char *);

    void ProfilingKernel();

    void DumpInfo();
    void CalcCM();
    void CalcCM0();
    void CalcCM(uint64);
    void CalcB2(float, float);
    void CalcB2(uint64, float, float);
    void SetEPS(float);
    void SetALLEPS2();
    void GetMortonOrder(uint64 *);

    void ConstructKeybasedTree(uint64, ArrayX3 &, ArrayX &);
    void CalcGravityOpenCL(ArrayX3 &, ArrayX &);
    void CalcGravity(ArrayX3 &, ArrayX &);
    void CalcRootBox(uint64, ArrayX3 &, ArrayX &);
    void SetParticles(uint64, ArrayX3 &, ArrayX &);

    void CalcKeys();
    void SortKeys();
    
    void ResetCounter() {
      count_interactions = 0.0;
    }

    double GetKernelTime(uint64 i) {
      return accum_ker_time[i]/count_force;
    }
    
    void ClearTimingInfo();
    void DumpTimingInfo();

    void SetVL(uint64);
    void EnableProfiling();
    void DisableProfiling();
    double GetCounter();
    void GetCM(Vector3 &);
    Vector4 GetCM();

    void CalcGravity(float &ppx, float &ppy, float &ppz, 
		     float &ax_t, float &ay_t, float &az_t, float &pot_t, 
		     uint64 &body, uint64 &cell, uint64 &ss);

    void CalcGravity(float &ppx, float &ppy, float &ppz);

  protected:
    uint64 MakeCell();
    bool CellOrNot(uint64);
    bool Particle(uint64);

    int Key(float);
    uint64 dilate3(uint32);
    uint32 dilate_3(unsigned short);
    uint64 poi(uint64, uint64);
    void Recursive(uint64 cur, uint64 level, uint64 nk, Keys *k);

    double GetKernelTime(cl::Event &e) {
      cl_ulong st = e.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      cl_ulong en = e.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      return (en - st)*1.0e-9;
    }

    uint64 nalloc, nalloc0;
    uint64 n, ncell;
    ArrayX4 B;
#define PX(i) B((i),0)
#define PY(i) B((i),1)
#define PZ(i) B((i),2)
#define MS(i) B((i),3)
    ArrayX4 X;
    ArrayX S;
    int *next, *more;
    Keys *kk;
    uint64 Root;
    float RootSize;

#define n_opencldevice0 4
    uint64 n_ocl_dev;
    uint64 n_per_device;
    std::vector<OpenCLDevice *> ov;
    cl::Kernel ker[n_opencldevice0];
    cl::Event ker_event[n_opencldevice0];
    cl::Buffer b_pos[n_opencldevice0], b_size[n_opencldevice0], b_next[n_opencldevice0], b_more[n_opencldevice0];
    cl::Buffer b_acc[n_opencldevice0], b_cc[n_opencldevice0];
    std::string kernel_options;

    uint64 opencl_offset[n_opencldevice0];
    uint64 opencl_grav_thread[n_opencldevice0];
    double accum_ker_time[n_opencldevice0];
    uint64 count_force;

    double kernel_execution_time;
    double count_interactions;

    float eps2, eps;
    uint64 VL_kernel;
    bool profiling;
    bool counter;

    uint64 maxdeeps;

    // parameters for tree
    static const uint64 empty;
    static const uint64 maxlevel;
    static const uint64 keyshift;
    static const float cell_mass_min_frac;
    static const float a_err_max_default;
    static const uint64 ncrit;

    cl::NDRange GetGlobalRange(uint64 __n) {
      return cl::NDRange(__n);
    }
    cl::NDRange GetLocalRange() {
      //      return cl::NDRange(2*8,8);
      //      return cl::NDRange(8*2,8);
      //      return cl::NDRange(16, 4);
      return cl::NDRange(64);
    }

  private:
    PHKeys PHK;
  };

  const uint64 GravityTree::empty = -1ULL;
  const uint64 GravityTree::maxlevel = 21;
  const uint64 GravityTree::keyshift = maxlevel/2;
  const float GravityTree::cell_mass_min_frac = 0.05;
  const float GravityTree::a_err_max_default = 0.01;
  const uint64 GravityTree::ncrit = 16;
  
  GravityTree::GravityTree(uint64 np) {
    nalloc0 = np;  
    nalloc  = 2*np;

    Allocate(nalloc,  B);
    Allocate(nalloc,  S);
    Allocate(nalloc0, X);

    next = new int[nalloc];
    more = new int[nalloc];
    kk   = new Keys[nalloc0];

    SetEPS(0.001);
    SetVL(8);
    DisableProfiling();
  }

  GravityTree::~GravityTree() {
    nalloc = 0;
    delete next;
    delete more;
    delete kk;
  }

  void GravityTree::SetEPS(float e)
  {
    eps  = e;
    eps2 = e*e;
  }

  void GravityTree::SetVL(uint64 n)
  {
    if (n <= 16) VL_kernel = n;
    else {
      std::cerr << "Unsupported VL\n";
      exit(-1);
    }
    std::stringstream options;
    if (VL_kernel != 0) {
      options << "-DVL=" << VL_kernel; 
      kernel_options = options.str();
    }
  }

  void GravityTree::EnableProfiling() {
    profiling = true;
    counter = true;
  }

  void GravityTree::DisableProfiling() {
    profiling = true;
    counter = false;
  }

  double GravityTree::GetCounter() {
    return count_interactions;
  }

  bool GravityTree::CellOrNot(uint64 ip)
  {
    return (ip >= n);
  }

  bool GravityTree::Particle(uint64 ip)
  {
    return !CellOrNot(ip);
  }

  uint64 GravityTree::MakeCell()
  {
    return (n + ncell++);
  }

  void GravityTree::CalcCM0()
  {
#pragma omp parallel for schedule(dynamic)
    for(uint64 i = 0; i < ncell; i++) {
      uint64 i_node, c_node, stop;
      Eigen::Vector4d cm;
      cm << 0.0, 0.0, 0.0, 0.0;

      i_node = Root + i;
      c_node = i_node;
      stop   = next[c_node];
      while(c_node != stop) {
	if ( Particle(c_node) ) {
	  cm.x() += MS(c_node)*PX(c_node);
	  cm.y() += MS(c_node)*PY(c_node);
	  cm.z() += MS(c_node)*PZ(c_node);
	  cm.w() += MS(c_node);
	  c_node = next[c_node];
	} else {
	  c_node = more[c_node];
	}
      }
      PX(i_node) = cm.x()/cm.w();
      PY(i_node) = cm.y()/cm.w();
      PZ(i_node) = cm.z()/cm.w();
      MS(i_node) = cm.w();
    }
  }

  void GravityTree::CalcCM(uint64 root)
  {
    uint64 c_node, stop;
    Eigen::Vector4d cm;
    cm << 0.0, 0.0, 0.0, 0.0;

    c_node = more[root];
    stop   = next[root];
    while(c_node != stop) {
      if ( !Particle(c_node) ) {
	CalcCM(c_node);
      }

      cm.x() += MS(c_node)*PX(c_node);
      cm.y() += MS(c_node)*PY(c_node);
      cm.z() += MS(c_node)*PZ(c_node);
      cm.w() += MS(c_node);

      c_node = next[c_node];
    }

    PX(root) = cm.x()/cm.w();
    PY(root) = cm.y()/cm.w();
    PZ(root) = cm.z()/cm.w();
    MS(root) = cm.w();
  }

  void GravityTree::CalcCM()
  {
    CalcCM(Root);
  }

  void GravityTree::CalcB2(uint64 root, float err, float m_frac)
  {
    uint64 i_node, stop;
    double b2, bmax2;
    float big_cell = 2.0*S[Root];

    b2 = 0.0;
    bmax2 = 0.0;

    i_node  = more[root];
    stop = next[root];
    while(i_node != stop) {
      if ( !Particle(i_node) ) {
	CalcB2(i_node, err, m_frac);
      }

      if (MS(root) > m_frac*MS(Root)) {
      } else {
	double dx, dy, dz, r2;
	dx = PX(root) - PX(i_node);
	dy = PY(root) - PY(i_node);
	dz = PZ(root) - PZ(i_node);
	r2 = dx*dx + dy*dy + dz*dz;
      
	bmax2 = std::max(bmax2, r2);
	b2 += MS(i_node)*r2;
      }

      i_node = next[i_node];
    }

    if (b2 == 0.0) {
      S[root] = big_cell;
    } else {
      S[root] = 0.5*sqrt(bmax2) + sqrt(0.25*bmax2 + sqrt(3.0*b2/err));
    }
    S[root] = S[root]*S[root];
  }

  void GravityTree::CalcB2(float err, float m_frac)
  {
    // make the Root node far too large.
    //    S[Root] *= 2.0;
    float big_cell = 2.0*S[Root];
#pragma omp parallel for schedule(dynamic)
    for(uint64 i = 0; i < ncell; i++) {
      uint64 i_node, c_node, stop;
      double b2, bmax2;

      b2 = 0.0;
      bmax2 = 0.0;

      i_node = Root+i;
      if (MS(i_node) > m_frac*MS(Root)) {
	// if a node is massive enough, make it larger too.
	S[i_node] = big_cell;
      } else {
	c_node = i_node;
	stop   = next[c_node];
	while(c_node != stop) {
	  if ( Particle(c_node) ) {
	    double dx, dy, dz, r2;
	    dx = PX(c_node) - PX(i_node);
	    dy = PY(c_node) - PY(i_node);
	    dz = PZ(c_node) - PZ(i_node);
	    r2 = dx*dx + dy*dy + dz*dz;
	    bmax2 = std::max(bmax2, r2);
	    b2 += MS(c_node)*r2;

	    c_node = next[c_node];
	  } else {
	    c_node = more[c_node];
	  }
	}
	// This is an absolute MAC.
	// See Salmon&Warren 1993 : Skeltons from the Treecode Closet
	S[i_node] = 0.5*sqrt(bmax2) + sqrt(0.25*bmax2 + sqrt(3.0*b2/err));
      }
      // compute the square of the node size 
      S[i_node] = S[i_node]*S[i_node];
    }
  }

  void GravityTree::SetALLEPS2()
  {
#pragma omp parallel for
    for(uint64 i = 0; i < n; i++) {
      S[i] = eps2;
    }
  }

  void GravityTree::DumpInfo(void)
  {
    std::cout << n << "\t" << ncell << "\t" << RootSize << "\n";
    std::cout << "root " << MS(Root) << " " << PX(Root) << " " << S[Root] << "\n";
  }

  int GravityTree::Key(float x)
  {
    return (int)(x*(float)(0x1<<maxlevel));
  }

  uint32 GravityTree::dilate_3(unsigned short t)
  {
    uint32 r = t;
    r = (r * 0x10001) & 0xFF0000FF;
    r = (r * 0x00101) & 0x0F00F00F;
    r = (r * 0x00011) & 0xC30C30C3;
    r = (r * 0x00005) & 0x49249249;
    return r;
  }

  uint64 GravityTree::dilate3(uint32 t)
  {
    uint64 h, l;
    uint64 mask = (0x1<<keyshift) - 1;

    h = (uint64)dilate_3( t >> keyshift );
    l = (uint64)dilate_3( t &  mask     );

    return h<<(3*keyshift) | l;
  }

  uint64 GravityTree::poi(uint64 key, uint64 level)
  {
    //    return (key >> 3*level)&0x7ULL;
    return PHK.poi(key, level);
  }

  void GravityTree::Recursive(uint64 cur, uint64 level, uint64 nk, Keys *k)
  {
    bool flag[8] = {false, false, false, false, false, false, false, false};
    uint64 cells[8], p_st[8], p_ne[8];

    if (level == 0) {
      std::cerr << " reach max level tree " <<"\n";
      exit(-1);
    }

    maxdeeps = std::min(maxdeeps, level);

    uint64 start = 0;
    for(uint64 i = 0; i < 8; i++) {
      uint64 j;
      if (i == 7) {
	// list all remaning particles
	j = nk;
      } else {
	// find a next bunch of particles with key == i
	if (poi(k[start].k, level) != i) {
	  // no particles
	  j = start;
	} else {
	  // use a binary search to find the boundary
	  uint64 lo = start;
	  uint64 hi = nk;
	  uint64 mid = (hi+lo)/2;
	  while(hi-lo != 1 && mid != hi) { 
	    uint64 q = poi(k[mid].k, level);
	    if (q > i) {
	      hi = mid;
	    } else {
	      lo = mid;
	    }
	    mid = (hi + lo)/2;
	  }

	  j = hi;
	}
      }
      uint64 ne = j - start;

      if (ne >= 1) {
	flag[i] = true;
	if (ne == 1) {
	  cells[i] = k[start].i;
	} else {
	  cells[i] = MakeCell();
	  //	  cell_level[cells[i]] = level;
	}
	p_st[i] = start;
	p_ne[i] = ne;
      }

      start = j;
    }

    uint64 fst_cell, lst_cell, c = 0, x[8];
    for(uint64 i = 0; i < 8; i++) {
      if (flag[i]) x[c++] = i;
    }
  
    fst_cell = cells[x[0]];
    if (c == 0) {
      lst_cell = fst_cell;
    } else {
      lst_cell = cells[x[c-1]];
    }

    for(uint64 i = 0; i < c; i++) {
      cells[i] = cells[x[i]];
      p_ne[i]  = p_ne[x[i]];
      p_st[i]  = p_st[x[i]];
    }

    if (fst_cell == lst_cell) {
      // one paticle
    } else {
      // link the cell to the next cell
      for(uint64 i = 0; i < c-1; i++) {
	next[cells[i]] = cells[i+1];
      }
    } 
    more[cur     ] = fst_cell;
    next[lst_cell] = next[cur];

    for(uint64 i = 0; i < c; i++) {
      if (p_ne[i] <= ncrit) {
	// stop recursion and link particles in a cells[i] 
	uint64 fst_cell, lst_cell;
	Keys *kk = &k[p_st[i]]; 
	uint64 np = p_ne[i];
	fst_cell = kk[0   ].i;
	lst_cell = kk[np-1].i;
	for(uint64 l = 0; l < np-1; l++) {
	  uint64 now = kk[l].i;
	  uint64 nei = kk[l+1].i;
	  next[now] = nei;
	}
	more[cells[i]] = fst_cell;
	next[lst_cell] = next[cells[i]];
      } else {
	// continue to build tree ...
	Recursive(cells[i], level-1, p_ne[i], &k[p_st[i]]);  
      }
    }
  }

  void GravityTree::GetMortonOrder(uint64 *list)
  {
    for(uint64 i = 0; i < n; i++) {
      list[i] = kk[i].i;
    }
  }

  void GravityTree::CalcKeys()
  {
    float cx, cy, cz;

    cx = PX(Root) - 0.5*RootSize;
    cy = PY(Root) - 0.5*RootSize;
    cz = PZ(Root) - 0.5*RootSize;

    float R_I = 1.0/RootSize;
#pragma omp parallel for
    for(uint64 i = 0; i < n; i++) {
      uint64 key;
      uint32 kx, ky, kz;
      kx = Key((PX(i) - cx)*R_I);
      ky = Key((PY(i) - cy)*R_I);
      kz = Key((PZ(i) - cz)*R_I);
      key = dilate3(kz)<<2 | dilate3(ky)<<1 | dilate3(kx);
      kk[i].k = key;
      kk[i].i = i;
      kk[i].k = PHK.Get(key, maxlevel);
    }
  }

  void GravityTree::SortKeys()
  {
#ifdef _OPENMP
    __gnu_parallel::sort(kk, kk + n, KeyCmp());
#else
    std::sort(kk, kk + n, KeyCmp());
#endif
  }

  void GravityTree::ProfilingKernel() {
    try {
      /*
      if (profiling) {
	cl_ulong st = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong en = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	kernel_execution_time = (en - st)*1.0e-9;
	std::cerr << kernel_execution_time << "\t";
	if (counter)  {
	  int *cc;
	  o->q.enqueueReadBuffer(b_cc, CL_TRUE, 0, n*sizeof(cl_int), X.data());
	  cc = (int *)X.data();
	  double cc_total = 0.0;
	  for(uint64 i = 0; i < n; i++) cc_total += (double)cc[i];
	  count_interactions += cc_total;
	  std::cerr << " " << 38.0*cc_total/kernel_execution_time/1.0e9 << "GFlops " << cc_total <<"\n";
	}
      }
      */
    }  catch( cl::Error e ) {
      std::cerr << e.what() << ":" << e.err() << "\n";
      exit(-1);
    }
  }

  void GravityTree::CalcGravity(float &ppx, float &ppy, float &ppz, 
				float &ax_t, float &ay_t, float &az_t, float &pot_t, 
				uint64 &body, uint64 &cell, uint64 &ss)
  {
    uint64 self;
    float dx, dy, dz, r2, r1i, r2i, phi;

    ax_t  = 0.0;
    ay_t  = 0.0;
    az_t  = 0.0;
    pot_t = 0.0;
    
    body = 0;
    cell = 0;
    ss = 0;

    self = Root;
    while(self != empty) {
      dx = ppx - PX(self);
      dy = ppy - PY(self);
      dz = ppz - PZ(self);
      r2 = dx*dx + dy*dy + dz*dz;
      
      if ( !CellOrNot(self) ) {
	if (r2 != 0.0) {
	  r2i   = 1.0/(r2 + S[self]);
	  r1i   = sqrt(r2i);
	  phi   = MS(self)*r1i;
	  pot_t-= phi;
	  phi  *= r2i;
	  ax_t -= phi*dx;
	  ay_t -= phi*dy;
	  az_t -= phi*dz;
	  body++;
	} else {
	  ss++;
	}
	self = next[self];
      } else {
	if (r2 > S[self]) {
	  r2i   = 1.0/r2;
	  r1i   = sqrt(r2i);
	  phi   = MS(self)*r1i;
	  pot_t-= phi;
	  phi  *= r2i;
	  ax_t -= phi*dx;
	  ay_t -= phi*dy;
	  az_t -= phi*dz;
	  cell++;
	  self = next[self];
	} else {
	  self = more[self];
	}
      }
    }
  }

  void GravityTree::CalcGravity(ArrayX3 &A, ArrayX &P)
  {
    uint64 count = 0;
    for(uint64 i = 0; i < n; i++) {
      uint64 body, cell, ss;
      float ppx, ppy, ppz, ax_t, ay_t, az_t, pot_t;

      ppx = PX(i);
      ppy = PY(i);
      ppz = PZ(i);

      GravityTree::CalcGravity(ppx, ppy, ppz, ax_t, ay_t, az_t, pot_t, body, cell, ss);

      A(i,0) = ax_t;
      A(i,1) = ay_t;
      A(i,2) = az_t;
      P(i)   = pot_t;

      count += body + cell;
    }
    std::cout << count << " " << (float)count/n << "\n";
  }

  void GravityTree::CalcGravityOpenCL(ArrayX3 &A, ArrayX &P)
  {
    try {
      int ns = n + ncell;

#pragma omp parallel for
      for(uint64 i = 0; i < n_ocl_dev; i++) {
        ker[i].setArg(0, b_pos[i]);
        ker[i].setArg(1, b_size[i]);
        ker[i].setArg(2, b_next[i]);
        ker[i].setArg(3, b_more[i]);
        ker[i].setArg(4, b_acc[i]);
        ker[i].setArg(5, (int)n);
        ker[i].setArg(6, (int)n);
        ker[i].setArg(7, (int)(opencl_offset[i]));

	cl_bool flag = CL_FALSE; 
	ov[i]->q.enqueueWriteBuffer(b_pos[i],  flag, 0, ns*sizeof(cl_float4), B.data());
	ov[i]->q.enqueueWriteBuffer(b_next[i], flag, 0, ns*sizeof(cl_int),    next);
	ov[i]->q.enqueueWriteBuffer(b_more[i], flag, 0, ns*sizeof(cl_int),    more);
	ov[i]->q.enqueueWriteBuffer(b_size[i], flag, 0, ns*sizeof(cl_float),  S.data());

        ov[i]->q.enqueueNDRangeKernel(ker[i],
				      cl::NullRange, 
                                      cl::NDRange(opencl_grav_thread[i]),
				      GetLocalRange(), 
				      NULL, &ker_event[i]);
        ov[i]->q.flush();
	ker_event[i].wait();

	flag = CL_TRUE;
	ov[i]->q.enqueueReadBuffer(b_acc[i], flag, 0, n_per_device*sizeof(cl_float4), 
				   X.bottomRows(n - opencl_offset[i]).data());

	accum_ker_time[i] += GetKernelTime(ker_event[i]);
      }

      A = X.leftCols(3);
      P = -X.rightCols(1);
    }  catch( cl::Error e ) {
      std::cerr << e.what() << ":" << e.err() << "\n";
      exit(-1);
    }

    count_force++;
  }

  void GravityTree::SetParticles(uint64 np, ArrayX3 &P, ArrayX &M)
  {
    // Block operations
    // See http://eigen.tuxfamily.org/dox/TutorialBlockOperations.html
    B.topLeftCorner(np,3)  = P;
    B.topRightCorner(np,1) = M;
  }

  void GravityTree::CalcRootBox(uint64 np, ArrayX3 &P, ArrayX &M)
  {
    n = np;
    SetParticles(np, P, M);

    Vector3 pmax = P.colwise().maxCoeff();
    Vector3 pmin = P.colwise().minCoeff();
    Vector3 box  = pmax - pmin;

    RootSize = box.maxCoeff();
    RootSize = RootSize*1.01;
    Vector3 center = 0.5*(pmax + pmin);

    ncell = 0;
    Root = MakeCell();
    S[Root] = RootSize;
    PX(Root) = center.x();
    PY(Root) = center.y();
    PZ(Root) = center.z();
    next[Root] = -1;
    //    cell_level[Root] = maxlevel;
    //    cell_level.assign(np, 0);
  }

  void GravityTree::ConstructKeybasedTree(uint64 np, ArrayX3 &P, ArrayX &M)
  {
    double dum_all = e_time();

    double dum = e_time();
    CalcRootBox(np, P, M);
    CalcKeys();
    std::cerr << " CK " << e_time() - dum;

    dum = e_time();
    SortKeys();

    std::cerr << " SK " << e_time() - dum;

    dum = e_time();
    maxdeeps = maxlevel-1;
    Recursive(Root, maxlevel-1, n, &kk[0]);
    std::cerr << " CT " << e_time() - dum;

    std::cerr << " SUM " <<  e_time() - dum_all << " " << maxdeeps << " :: "; 
  }

  void GravityTree::GetCM(Vector3 &c)
  {
    c.x() = PX(Root);
    c.y() = PY(Root);
    c.z() = PZ(Root);
  }

  Vector4 GravityTree::GetCM()
  {
    return B.row(Root);
  }

  void GravityTree::SetupOpenCL(int ip = 0, int id = 0, const char *k_f = "grav.cl")
  {
    try {
      if (id < 0) {
	id = -id;
	n_ocl_dev = id;
      } else {
	n_ocl_dev = 1;   
      }
      n_per_device = nalloc0/n_ocl_dev;
      for(uint64 i = 0; i < n_ocl_dev; i++) {
	ov.push_back(new OpenCLDevice(ip));    
	ov[i]->SetupContext( n_ocl_dev == 1 ? id :  i );

	ov[i]->SetKernelOptions(kernel_options);
	ov[i]->BuildOpenCLKernels(k_f);
	ker[i] = ov[i]->GetKernel("tree_v");

	// read only buffers (HOST -> GPU)
	b_pos[i]  = cl::Buffer(ov[i]->ctx, CL_MEM_READ_ONLY, nalloc*sizeof(cl_float4));
	b_size[i] = cl::Buffer(ov[i]->ctx, CL_MEM_READ_ONLY, nalloc*sizeof(cl_float));
	b_next[i] = cl::Buffer(ov[i]->ctx, CL_MEM_READ_ONLY, nalloc*sizeof(cl_int));
	b_more[i] = cl::Buffer(ov[i]->ctx, CL_MEM_READ_ONLY, nalloc*sizeof(cl_int));

	// output buffers (GPU -> HOST)
	b_acc[i]  = cl::Buffer(ov[i]->ctx, CL_MEM_WRITE_ONLY, (n_per_device)*sizeof(cl_float4));

	opencl_offset[i]      = n_per_device*i;
	opencl_grav_thread[i] = n_per_device/VL_kernel;
      }
    }  catch( cl::Error e ) {
      std::cerr << e.what() << ":" << e.err() << "\n";
      exit(-1);
    }
  }

  void GravityTree::ClearTimingInfo()
  {
    for(uint64 i = 0; i < n_opencldevice0; i++) {
      accum_ker_time[i] = 0.0;
    }
    count_force = 0; 
  }

  void GravityTree::DumpTimingInfo()
  {
    std::cerr << "GRAV ";
    for(uint64 i = 0; i < n_ocl_dev; i++) {
      std::cerr << i << " " << accum_ker_time[i]/count_force << "\t";
    }
    std::cerr << "\n";
  }
}
#endif
