#ifndef OTOO_MODULE_H
#define OTOO_MODULE_H
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <parallel/algorithm>

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace OTOO {
  typedef unsigned long long int uint64;
  typedef unsigned int           uint32;
  typedef Eigen::Vector3f  Vector3;
  typedef Eigen::Vector4f  Vector4;
  typedef Eigen::ArrayX3f  ArrayX3;
  typedef Eigen::ArrayX4f  ArrayX4;
  typedef Eigen::ArrayX2f  ArrayX2;
  typedef Eigen::ArrayXf   ArrayX;

  double e_time(void)
  {
    static struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec  + now.tv_usec/1000000.0); 
  }

  template<typename AR>
  void Allocate(const uint64 n, AR &x)
  {
    x.resize(n, Eigen::NoChange);
  }

  typedef struct keys {
    uint64 k, i;
  } Keys;

  struct KeyCmp {
    bool operator()(const keys &a, const keys &b) const {
      return a.k < b.k;
    }
  };
}
#endif
