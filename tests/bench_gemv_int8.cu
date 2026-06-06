/* bench_gemv_int8.cu — BETA.3a-v3 isolated GEMV crossover sweep.
 *
 * Strips away attention, argmax, KV, and the ~250-launch decode overhead to
 * isolate the ONE thing the dp4a kernel optimizes: the weight matmul. Sweeps the
 * weight-matrix dimension N (single-token GEMV, M=1, y[N] = W[N,N] . x[N]) and
 * compares:
 *    f32  : cuBLAS SGEMV  (reads N*N * 4 bytes of weights)
 *    int8 : dp4a GEMV     (reads N*N * 1 byte  of weights, the v2 warp-per-row
 *                          kernel — identical to cuda_forward.cu k_gemv_q8_dp4a_v2)
 *
 * At small N both are overhead/compute-bound (ratio ~1). As N grows the 336 GB/s
 * GDDR6 bus saturates; f32 (4 B/weight) chokes first and int8 (1 B/weight) pulls
 * ahead toward ~4x. The crossover is the bus-saturation point.
 *
 * Build (host, CUDA toolkit on PATH):
 *   nvcc -O3 -arch=sm_75 bench_gemv_int8.cu -lcublas -o bench_gemv_int8
 * Run with BOTH clocks pinned:
 *   nvidia-smi -lgc 1500,1500 && nvidia-smi -lmc 7000   (then -rgc / -rmc after)
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

/* --- identical to cuda_forward.cu --- */
__global__ void k_quant_act_int8(const float *x, int n, int npad,
                                 signed char *qx, float *sx) {
    __shared__ float sm[256];
    float m = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float a = fabsf(x[i]); if (a > m) m = a; }
    sm[threadIdx.x] = m; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o && sm[threadIdx.x + o] > sm[threadIdx.x]) sm[threadIdx.x] = sm[threadIdx.x + o];
        __syncthreads();
    }
    float scale = sm[0] > 0.0f ? sm[0] * (1.0f / 127.0f) : 1.0f;
    if (threadIdx.x == 0) *sx = scale;
    float inv = 1.0f / scale;
    for (int i = threadIdx.x; i < npad; i += blockDim.x) {
        float v = (i < n) ? x[i] * inv : 0.0f;
        int q = __float2int_rn(v); if (q > 127) q = 127; if (q < -127) q = -127;
        qx[i] = (signed char)q;
    }
}
__global__ void k_gemv_q8_dp4a_v2(const signed char *codes, const unsigned long long *row_off,
                                  const float *row_scale, int in, const signed char *qx,
                                  const float *sx, float *y, int out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= out) return;
    const int4 *wrow = (const int4 *)(codes + row_off[j]);
    const int4 *qxi  = (const int4 *)qx;
    const int n16 = in >> 4;
    int acc = 0;
    for (int c = lane; c < n16; c += 32) {
        int4 wv = wrow[c], qv = qxi[c];
        acc = __dp4a(wv.x, qv.x, acc);
        acc = __dp4a(wv.y, qv.y, acc);
        acc = __dp4a(wv.z, qv.z, acc);
        acc = __dp4a(wv.w, qv.w, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[j] = (float)acc * (row_scale[j] * (1.0f / 127.0f)) * (*sx);
}

/* BETA.3a-v4: Q4 dp4a GEMV. Arena Q4 = 2 nibbles/byte (low=even idx, high=odd),
 * sign-extended (n^8)-8, qmax=7. Read int4 (16 B = 32 packed weights) straight
 * from VRAM, unpack the nibbles to int8 IN THE ALU (free under memory-bound),
 * feed dp4a. Activation stays int8 (qmax 127). `in` must be a multiple of 32.
 * 0.5 B/weight => theoretical 8:1 over f32 (2x the int8 path). */
__device__ __forceinline__ void unpack8(int w, int &lo4, int &hi4) {
    int b0=w&0xFF, b1=(w>>8)&0xFF, b2=(w>>16)&0xFF, b3=(w>>24)&0xFF;
    #define SX(byte,hi) (((((((hi)?((byte)>>4):(byte)))&0xF)^0x8)-0x8)&0xFF)
    lo4 = SX(b0,0) | (SX(b0,1)<<8) | (SX(b1,0)<<16) | (SX(b1,1)<<24);
    hi4 = SX(b2,0) | (SX(b2,1)<<8) | (SX(b3,0)<<16) | (SX(b3,1)<<24);
    #undef SX
}
__global__ void k_gemv_q4_dp4a_v2(const unsigned char *codes, const unsigned long long *row_off,
                                  const float *row_scale, int in, const signed char *qx,
                                  const float *sx, float *y, int out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= out) return;
    const int4 *wrow = (const int4 *)(codes + row_off[j]);   /* 16 B = 32 Q4 weights */
    const int4 *qxi  = (const int4 *)qx;
    const int n32 = in >> 5;
    int acc = 0, a, b;
    for (int c = lane; c < n32; c += 32) {
        int4 wv = wrow[c];
        int4 q0 = qxi[2*c], q1 = qxi[2*c + 1];               /* 32 matching int8 acts */
        unpack8(wv.x, a, b); acc = __dp4a(a, q0.x, acc); acc = __dp4a(b, q0.y, acc);
        unpack8(wv.y, a, b); acc = __dp4a(a, q0.z, acc); acc = __dp4a(b, q0.w, acc);
        unpack8(wv.z, a, b); acc = __dp4a(a, q1.x, acc); acc = __dp4a(b, q1.y, acc);
        unpack8(wv.w, a, b); acc = __dp4a(a, q1.z, acc); acc = __dp4a(b, q1.w, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[j] = (float)acc * (row_scale[j] * (1.0f / 7.0f)) * (*sx);
}

static double time_kernel(void (*launch)(void*), void *ctx, int iters) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 10; i++) launch(ctx);              /* warm */
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch(ctx);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return (double)ms / iters;                              /* ms per GEMV */
}

struct F32Ctx { cublasHandle_t h; const float *W, *x; float *y; int N; };
static void launch_f32(void *p) {
    F32Ctx *c = (F32Ctx *)p; const float a = 1.0f, b = 0.0f;
    /* y[N] = W[N(out) x N(in)] . x[N]; W row-major => use SGEMV with op_T on column-major. */
    cublasSgemv(c->h, CUBLAS_OP_T, c->N, c->N, &a, c->W, c->N, c->x, 1, &b, c->y, 1);
}
struct I8Ctx { const signed char *codes; const unsigned long long *roff; const float *rscale;
               const signed char *qx; const float *sx; const float *x; float *y; int N; };
static void launch_i8(void *p) {
    I8Ctx *c = (I8Ctx *)p;
    int npad = (c->N + 15) & ~15;
    k_quant_act_int8<<<1, 256, 0, 0>>>(c->x, c->N, npad, (signed char *)c->qx, (float *)c->sx);
    unsigned blocks = ((unsigned)c->N + 7u) / 8u;
    k_gemv_q8_dp4a_v2<<<blocks, 256, 0, 0>>>(c->codes, c->roff, c->rscale, c->N, c->qx, c->sx, c->y, c->N);
}
struct Q4Ctx { const unsigned char *codes; const unsigned long long *roff; const float *rscale;
               const signed char *qx; const float *sx; const float *x; float *y; int N; };
static void launch_q4(void *p) {
    Q4Ctx *c = (Q4Ctx *)p;
    int npad = (c->N + 31) & ~31;                            /* 32-weight chunks */
    k_quant_act_int8<<<1, 256, 0, 0>>>(c->x, c->N, npad, (signed char *)c->qx, (float *)c->sx);
    unsigned blocks = ((unsigned)c->N + 7u) / 8u;
    k_gemv_q4_dp4a_v2<<<blocks, 256, 0, 0>>>(c->codes, c->roff, c->rscale, c->N, c->qx, c->sx, c->y, c->N);
}

int main(int argc, char **argv) {
    int dev = 0; CK(cudaSetDevice(dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    double peakGBs = 336.0;   /* RTX 2060: GDDR6 14 Gbps x 192-bit = 336 GB/s (peak) */
    printf("# device: %s  sm_%d%d  peak ~%.0f GB/s (GDDR6 192-bit)\n",
           prop.name, prop.major, prop.minor, peakGBs);
    printf("# %6s | %9s %9s %7s | %9s %9s %7s | %9s\n",
           "N", "f32 us", "int8 us", "i8 x", "q4 us", "q4 x", "q4 GB/s", "f32 GB/s");

    cublasHandle_t h; cublasCreate(&h);
    int Ns[] = { 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384 };
    int iters = 200;

    for (size_t k = 0; k < sizeof(Ns)/sizeof(Ns[0]); k++) {
        int N = Ns[k];
        size_t WN = (size_t)N * N;
        /* host weights: small random, then int8-quantize per row (sym, /127). */
        std::vector<float> hW(WN);
        std::vector<signed char> hC(WN);         /* Q8 codes */
        std::vector<unsigned char> hC4(WN/2);    /* Q4 packed: 2 nibbles/byte */
        std::vector<float> hScale(N), hX(N);
        std::vector<unsigned long long> hOff(N), hOff4(N);
        for (int j = 0; j < N; j++) {
            float mx = 0;
            for (int i = 0; i < N; i++) { float v = ((float)((j*131+i*17)%255)/255.f-0.5f); hW[(size_t)j*N+i]=v; if(fabsf(v)>mx)mx=fabsf(v);}
            hScale[j] = mx > 0 ? mx : 1.f;
            hOff[j]  = (unsigned long long)j * N;
            hOff4[j] = (unsigned long long)j * (N/2);
            for (int i = 0; i < N; i++) { int q=(int)lrintf(hW[(size_t)j*N+i]/hScale[j]*127.f); if(q>127)q=127; if(q<-127)q=-127; hC[(size_t)j*N+i]=(signed char)q; }
            for (int i = 0; i < N; i += 2) {     /* Q4: low=even, high=odd, nib=q&0xF */
                int q0=(int)lrintf(hW[(size_t)j*N+i]/hScale[j]*7.f);   if(q0>7)q0=7; if(q0<-7)q0=-7;
                int q1=(int)lrintf(hW[(size_t)j*N+i+1]/hScale[j]*7.f); if(q1>7)q1=7; if(q1<-7)q1=-7;
                hC4[((size_t)j*N + i) >> 1] = (unsigned char)((q0 & 0xF) | ((q1 & 0xF) << 4));
            }
        }
        for (int i = 0; i < N; i++) hX[i] = ((float)((i*7)%101)/101.f - 0.5f);

        float *dW=nullptr,*dx=nullptr,*dy=nullptr,*dscale=nullptr,*dsx=nullptr;
        signed char *dC=nullptr,*dqx=nullptr; unsigned char *dC4=nullptr;
        unsigned long long *dOff=nullptr,*dOff4=nullptr;
        CK(cudaMalloc(&dW, WN*sizeof(float)));
        CK(cudaMalloc(&dC, WN));
        CK(cudaMalloc(&dC4, WN/2));
        CK(cudaMalloc(&dx, (size_t)N*sizeof(float)));
        CK(cudaMalloc(&dy, (size_t)N*sizeof(float)));
        CK(cudaMalloc(&dscale, (size_t)N*sizeof(float)));
        CK(cudaMalloc(&dOff, (size_t)N*sizeof(unsigned long long)));
        CK(cudaMalloc(&dOff4, (size_t)N*sizeof(unsigned long long)));
        CK(cudaMalloc(&dqx, (size_t)((N+31)&~31)));
        CK(cudaMalloc(&dsx, sizeof(float)));
        CK(cudaMemcpy(dW, hW.data(), WN*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dC, hC.data(), WN, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dC4, hC4.data(), WN/2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, hX.data(), (size_t)N*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dscale, hScale.data(), (size_t)N*sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dOff, hOff.data(), (size_t)N*sizeof(unsigned long long), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dOff4, hOff4.data(), (size_t)N*sizeof(unsigned long long), cudaMemcpyHostToDevice));

        F32Ctx fc{ h, dW, dx, dy, N };
        I8Ctx  ic{ dC, dOff, dscale, dqx, dsx, dx, dy, N };
        Q4Ctx  qc{ dC4, dOff4, dscale, dqx, dsx, dx, dy, N };
        double f32ms = time_kernel(launch_f32, &fc, iters);
        double i8ms  = time_kernel(launch_i8,  &ic, iters);
        double q4ms  = time_kernel(launch_q4,  &qc, iters);

        /* CORRECTNESS (first N only): Q4 device output == host reference using the
         * SAME device-quantized activation + same nibble decode. Proves the unpack. */
        if (k == 0) {
            launch_q4(&qc); CK(cudaDeviceSynchronize());
            std::vector<float> hy(N); std::vector<signed char> hqx((N+31)&~31); float hsx=0;
            CK(cudaMemcpy(hy.data(), dy, (size_t)N*sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hqx.data(), dqx, (size_t)((N+31)&~31), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&hsx, dsx, sizeof(float), cudaMemcpyDeviceToHost));
            double maxrel = 0;
            for (int j = 0; j < N; j++) {
                long acc = 0;
                for (int i = 0; i < N; i++) {
                    unsigned char byte = hC4[((size_t)j*N + i) >> 1];
                    int nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                    int code = (nib ^ 8) - 8;
                    acc += (long)code * (long)hqx[i];
                }
                double ref = (double)acc * (hScale[j] / 7.0) * (double)hsx;
                double d = fabs((double)hy[j] - ref), denom = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
                if (d/denom > maxrel) maxrel = d/denom;
            }
            printf("# Q4 correctness @N=%d: max rel err %.2e vs host ref  -> %s\n",
                   N, maxrel, maxrel < 1e-4 ? "PASS" : "FAIL");
        }

        double f32GBs = (double)WN * 4.0 / (f32ms * 1e-3) / 1e9;
        double q4GBs  = (double)WN * 0.5 / (q4ms  * 1e-3) / 1e9;
        printf("  %6d | %8.1f %8.1f %6.2fx | %8.1f %6.2fx %8.0f | %8.0f\n",
               N, f32ms*1e3, i8ms*1e3, f32ms/i8ms, q4ms*1e3, f32ms/q4ms, q4GBs, f32GBs);

        cudaFree(dW); cudaFree(dC); cudaFree(dC4); cudaFree(dx); cudaFree(dy);
        cudaFree(dscale); cudaFree(dOff); cudaFree(dOff4); cudaFree(dqx); cudaFree(dsx);
    }
    cublasDestroy(h);
    return 0;
}
