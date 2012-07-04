#define READONLY_P const * restrict 

#ifdef __GPU__
#define my_rsqrt native_rsqrt
#define my_sqrt  native_sqrt
#define my_recip native_recip
#else
#define my_rsqrt rsqrt
#define my_sqrt  sqrt
float my_recip(float q) {
  return 1.0f/q;
}
#endif

float R2(float4 p)
{
  return p.x*p.x + p.y*p.y + p.z*p.z;
}

float min2(float *x)
{
  return min(x[0], x[1]);
}

float min4(float *x)
{
  return min(min2(&x[0]), min2(&x[2]));
}

float min8(float *x)
{
  return min(min4(&x[0]), min4(&x[4]));
}

float min16(float *x)
{
  return min(min8(&x[0]), min8(&x[8]));
}

void g2(float4 dx, float r2, float mj, float4 *a)
{
#ifdef __GPU__
  float r1i = native_rsqrt(r2);
#else
  float r1i = rsqrt(r2);
#endif
  float r2i = r1i*r1i;
  float r1im = mj*r1i;
  float r3im = r1im*r2i;
  float4 f = (float4)(r3im, r3im, r3im, r1im);

  dx.w = 1.0f;
  *a = mad(dx, f, *a);
}

float minn(float *x)
{
#if VL == 1
  return x[0];
#endif
#if VL == 2
  return min2(x);
#endif
#if VL == 4
  return min4(x);
#endif
#if VL == 8
  return min8(x);
#endif
#if VL == 16
  return min16(x);
#endif
}

__kernel
void 
tree_v(__global float4 READONLY_P pos, 
       __global float READONLY_P size,
       __global int READONLY_P next,
       __global int READONLY_P more, 
       __global float4 *acc_g,
       const int root, 
       const int n,
       const int offset
#ifdef COUNT
       , __global int *counter
#endif
)
{
  unsigned int g_xid = get_global_id(0);
  unsigned int g_yid = get_global_id(1);
  unsigned int g_w   = get_global_size(0);
  unsigned int gid   = g_yid*g_w + g_xid;
  //  gid = get_global_id(0);
  unsigned int base0 = gid*VL;
  //  unsigned int base  = (gid+offset)*VL;
  unsigned int base  = base0 + offset;

  float4 p[VL];
  float4 acc[VL];
#ifdef COUNT
  int cc = 0;
#endif

  for(int i = 0; i < VL; i++) {
    p[i] = pos[base + i];
    acc[i] = (float4)(0.0f);
  }

  int cur = more[n];
  while(cur != -1) {
    float4 q = pos[cur];
    float  mj = q.w;
    float  s = size[cur];
    float4 dx[VL];
    float r2[VL];

    for(int i = 0; i < VL; i++) {
      dx[i] = q - p[i];
      r2[i] = R2(dx[i]);
    }

    if (cur < n || s < minn(r2)) {
      float e2 = (cur < n) ? s : 0.0f;
      for(int i = 0; i < VL; i++) {
	float mjj;
	mjj = (base+i != cur) ? mj : 0.0f;
	g2(dx[i], r2[i]+e2, mjj, &acc[i]);
      }
#ifdef COUNT
      cc++;
#endif
      cur = next[cur];
    } else {
      cur = more[cur];
    }
  }

  for(int i = 0; i < VL; i++) {
    acc_g[base0+i] = (float4)acc[i];
#ifdef COUNT
    counter[base0+i] = cc;
#endif
  }
}
