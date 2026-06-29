/* sp_eagle_fwd_cuda.cu — CUDA reference of the gemma4-assistant (EAGLE/MTP) draft forward.
 *
 * Serving step: ports the proven draft forward (sp_eagle_fwd.c / G-EAGLE-DRAFT-FWD-C GREEN,
 * itself == the numpy oracle) to CUDA kernels, gated NUMERICALLY against the SAME oracle
 * fixture (dumped weights + inputs + expected post-output_norm hidden) — so the CUDA draft
 * math is validated WITHOUT the live 12B. Self-contained: loads raw-f32 weights/inputs, runs
 * naive double-accum kernels, compares the final hidden + h_next. The vocab head is a plain
 * matmul (validated by the CPU gate) and is intentionally skipped here.
 *
 * In the served daemon these kernels read the resident sp_g4_kv KV ring (dKc[NL-1]/[NL-2]) and
 * the feature from gemma4_kv_capture_feat; here K/V/feature come from the fixture (same numbers).
 *
 * Build:  nvcc -O2 -arch=sm_75 sp_eagle_fwd_cuda.cu -o sp_eagle_fwd_cuda.exe
 * Usage:  sp_eagle_fwd_cuda <fixture_dir>
 * Gate G-EAGLE-DRAFT-FWD-CUDA: hidden max|d|<1e-2 && relL2<1e-3 (+ h_next).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define NH 16
#define HID 1024
#define BB 3840
#define FF 8192
#define EPSF 1e-6f
#define P 12
#define POS 7
static const int SWAp[4] = {1, 1, 1, 0};
#define CK(e) do{ cudaError_t _e=(e); if(_e!=cudaSuccess){ fprintf(stderr,"CUDA %s @ %d: %s\n",#e,__LINE__,cudaGetErrorString(_e)); exit(2);} }while(0)

/* y[r] = sum_c W[r*cols+c]*x[c]  (double accum, one thread per output row) */
__global__ void k_matmul(const float *W, const float *x, float *y, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *w = W + (size_t)r * cols; double a = 0;
    for (int c = 0; c < cols; c++) a += (double)w[c] * x[c];
    y[r] = (float)a;
}
/* out[i] = x[i]*inv*g[i], inv=1/sqrt(mean(x^2)+eps), over d (single block reduce) */
__global__ void k_rms(const float *x, const float *g, float *out, int d) {
    __shared__ double sh[256];
    double s = 0; for (int i = threadIdx.x; i < d; i += blockDim.x) s += (double)x[i] * x[i];
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    float inv = (float)(1.0 / sqrt(sh[0] / d + EPSF));
    for (int i = threadIdx.x; i < d; i += blockDim.x) out[i] = x[i] * inv * g[i];
}
/* per-head RMSNorm (no rope): q[NH*hd], g[hd]; one block per head */
__global__ void k_rms_head(float *q, const float *g, int hd) {
    int h = blockIdx.x; float *qh = q + (size_t)h * hd;
    __shared__ double sh[256];
    double s = 0; for (int i = threadIdx.x; i < hd; i += blockDim.x) s += (double)qh[i] * qh[i];
    sh[threadIdx.x] = s; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    float inv = (float)(1.0 / sqrt(sh[0] / hd + EPSF));
    for (int i = threadIdx.x; i < hd; i += blockDim.x) qh[i] = qh[i] * inv * g[i];
}
/* neox rope per head: rotate (i, i+hd/2); one thread per (head,i<half) */
__global__ void k_rope(float *q, int hd, int pos, double base) {
    int h = blockIdx.x, i = threadIdx.x, half = hd / 2;
    if (i >= half) return;
    float *qh = q + (size_t)h * hd;
    double inv = pow(base, -(2.0 * i) / hd), c = cos(pos * inv), s = sin(pos * inv);
    float a = qh[i], b = qh[i + half];
    qh[i] = (float)(a * c - b * s); qh[i + half] = (float)(b * c + a * s);
}
/* Q-only attention over fixture K/V [P,hd]; one block per head -> ctx[NH*hd] */
__global__ void k_attn(const float *q, const float *K, const float *V, float *ctx, int hd) {
    int h = blockIdx.x; const float *qh = q + (size_t)h * hd; float *ch = ctx + (size_t)h * hd;
    float asc = 1.0f / sqrtf((float)hd);
    __shared__ float sc[P]; __shared__ float mx, sum;
    if (threadIdx.x < P) { double s = 0; const float *kt = K + (size_t)threadIdx.x * hd;
        for (int d = 0; d < hd; d++) s += (double)kt[d] * qh[d]; sc[threadIdx.x] = (float)s * asc; }
    __syncthreads();
    if (threadIdx.x == 0) { mx = sc[0]; for (int t = 1; t < P; t++) if (sc[t] > mx) mx = sc[t];
        sum = 0; for (int t = 0; t < P; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; } }
    __syncthreads();
    for (int d = threadIdx.x; d < hd; d += blockDim.x) { double a = 0;
        for (int t = 0; t < P; t++) a += (double)sc[t] * V[(size_t)t * hd + d]; ch[d] = (float)(a / sum); }
}
__global__ void k_gelu_mul(float *g, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float x = g[i]; g[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))) * up[i];
}
__global__ void k_add(float *x, const float *y, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=y[i]; }
__global__ void k_axs(float *x, const float *s, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]*=s[0]; }
__global__ void k_addv(float *o, const float *a, const float *b, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]+b[i]; }

static long fsize_floats(const char *p){ FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"no %s\n",p);exit(2);} fseek(f,0,SEEK_END); long n=ftell(f)/4; fclose(f); return n; }
static float *dload(const char *dir, const char *nm, long *n_out){
    char p[1024]; snprintf(p,sizeof p,"%s/%s",dir,nm);
    long n = fsize_floats(p); FILE*f=fopen(p,"rb"); float*h=(float*)malloc(n*4);
    if(fread(h,4,n,f)!=(size_t)n){fprintf(stderr,"short %s\n",p);exit(2);} fclose(f);
    float*d; CK(cudaMalloc(&d,n*4)); CK(cudaMemcpy(d,h,n*4,cudaMemcpyHostToDevice)); free(h);
    if(n_out)*n_out=n; return d;
}
static float *dloadw(const char *dir, const char *nm){ char p[256]; snprintf(p,sizeof p,"w/%s",nm); return dload(dir,p,0); }

#define GRID(n) (unsigned)(((n)+255)/256)

int main(int argc, char **argv) {
    const char *fix = argc > 1 ? argv[1] : "fixture";
    long n;
    float *pre = dloadw(fix,"pre_proj.f32"), *post = dloadw(fix,"post_proj.f32"), *onorm = dloadw(fix,"out_norm.f32");
    float *x = dload(fix,"x.f32",0), *h0 = dload(fix,"h.f32",0);
    float *xh; CK(cudaMalloc(&xh, 2*BB*4));
    CK(cudaMemcpy(xh, x, BB*4, cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(xh+BB, h0, BB*4, cudaMemcpyDeviceToDevice));
    float *cur; CK(cudaMalloc(&cur, HID*4));
    k_matmul<<<GRID(HID),256>>>(pre, xh, cur, HID, 2*BB);                 /* pre_proj -> 1024 */

    float *nbuf,*abuf,*attn_out,*q,*ctx,*gb,*ub;
    CK(cudaMalloc(&nbuf,HID*4)); CK(cudaMalloc(&abuf,HID*4)); CK(cudaMalloc(&attn_out,HID*4));
    CK(cudaMalloc(&q,NH*512*4)); CK(cudaMalloc(&ctx,NH*512*4)); CK(cudaMalloc(&gb,FF*4)); CK(cudaMalloc(&ub,FF*4));
    printf("[chain] pre_proj=%d", HID);
    for (int il = 0; il < 4; il++) {
        char b[32]; snprintf(b,sizeof b,"l%d_",il);
        char nm[40];
        #define LW(t) (snprintf(nm,sizeof nm,"%s%s.f32",b,t), dloadw(fix,nm))
        float *attn_norm=LW("attn_norm"), *wq=LW("wq"), *qn=LW("qn"), *wo=LW("wo"), *pan=LW("post_attn"),
              *ffn_norm=LW("ffn_norm"), *wg=LW("wg"), *wu=LW("wu"), *wd=LW("wd"), *pfn=LW("post_ffw"), *osc=LW("osc");
        char wqn[40]; snprintf(wqn,sizeof wqn,"w/l%d_wq.f32",il); char wqp[1024]; snprintf(wqp,sizeof wqp,"%s/%s",fix,wqn);
        long qd = fsize_floats(wqp)/HID; int hd = (int)qd / NH;
        char kn[32],vn[32]; snprintf(kn,sizeof kn,"k%d.f32",il); snprintf(vn,sizeof vn,"v%d.f32",il);
        float *K=dload(fix,kn,&n), *V=dload(fix,vn,0);
        double base = SWAp[il] ? 1e4 : 1e6;
        k_rms<<<1,256>>>(cur, attn_norm, nbuf, HID);
        k_matmul<<<GRID(qd),256>>>(wq, nbuf, q, (int)qd, HID);
        k_rms_head<<<NH,256>>>(q, qn, hd);
        k_rope<<<NH, hd/2>>>(q, hd, POS, base);
        k_attn<<<NH,256>>>(q, K, V, ctx, hd);
        k_matmul<<<GRID(HID),256>>>(wo, ctx, abuf, HID, (int)qd);
        k_rms<<<1,256>>>(abuf, pan, nbuf, HID);
        k_addv<<<GRID(HID),256>>>(attn_out, nbuf, cur, HID);
        k_rms<<<1,256>>>(attn_out, ffn_norm, nbuf, HID);
        k_matmul<<<GRID(FF),256>>>(wg, nbuf, gb, FF, HID);
        k_matmul<<<GRID(FF),256>>>(wu, nbuf, ub, FF, HID);
        k_gelu_mul<<<GRID(FF),256>>>(gb, ub, FF);
        k_matmul<<<GRID(HID),256>>>(wd, gb, nbuf, HID, FF);
        k_rms<<<1,256>>>(nbuf, pfn, abuf, HID);
        k_addv<<<GRID(HID),256>>>(cur, abuf, attn_out, HID);
        k_axs<<<GRID(HID),256>>>(cur, osc, HID);
        printf(" -> blk.%d(hd=%d)=%d", il, hd, HID);
        cudaFree(attn_norm);cudaFree(wq);cudaFree(qn);cudaFree(wo);cudaFree(pan);cudaFree(ffn_norm);
        cudaFree(wg);cudaFree(wu);cudaFree(wd);cudaFree(pfn);cudaFree(osc);cudaFree(K);cudaFree(V);
        #undef LW
    }
    k_rms<<<1,256>>>(cur, onorm, nbuf, HID);                              /* output_norm -> final hidden */
    float *hnext; CK(cudaMalloc(&hnext, BB*4));
    k_matmul<<<GRID(BB),256>>>(post, nbuf, hnext, BB, HID);
    CK(cudaDeviceSynchronize());
    printf(" -> output_norm=%d -> h_next=%d\n", HID, BB);

    float hid_c[HID], hn_c[BB];
    CK(cudaMemcpy(hid_c, nbuf, HID*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hn_c, hnext, BB*4, cudaMemcpyDeviceToHost));
    /* expected from oracle fixture */
    float *eh=(float*)malloc(HID*4), *ehn=(float*)malloc(BB*4);
    { char p[1024]; snprintf(p,sizeof p,"%s/hidden.f32",fix); FILE*f=fopen(p,"rb"); fread(eh,4,HID,f); fclose(f);
      snprintf(p,sizeof p,"%s/hnext.f32",fix); f=fopen(p,"rb"); fread(ehn,4,BB,f); fclose(f); }
    double hm=0,hne=0,hde=0; for(int i=0;i<HID;i++){double d=(double)hid_c[i]-eh[i]; if(fabs(d)>hm)hm=fabs(d); hne+=d*d; hde+=(double)eh[i]*eh[i];}
    double nm2=0,nne=0,nde=0; for(int i=0;i<BB;i++){double d=(double)hn_c[i]-ehn[i]; if(fabs(d)>nm2)nm2=fabs(d); nne+=d*d; nde+=(double)ehn[i]*ehn[i];}
    double hrel=sqrt(hne/(hde+1e-12)), nrel=sqrt(nne/(nde+1e-12));
    int ok = hm<1e-2 && hrel<1e-3 && nm2<1e-2 && nrel<1e-3;
    printf("[gate] hidden max|d|=%.4g relL2=%.3g | h_next max|d|=%.4g relL2=%.3g\n", hm,hrel,nm2,nrel);
    printf("G-EAGLE-DRAFT-FWD-CUDA: %s\n", ok?"GREEN":"RED");
    return ok?0:1;
}
