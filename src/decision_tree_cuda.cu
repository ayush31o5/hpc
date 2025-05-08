#include <cuda_runtime.h>
#include <float.h>
#include <math.h> // for log2f()
#include "decision_tree.h"

// Device helper to compute entropy
__device__ inline float entropy(int c0, int c1)
{
    int N = c0 + c1;
    if (N == 0)
        return 0.0f;
    float p0 = c0 / float(N), p1 = c1 / float(N);
    float e = 0.0f;
    if (p0 > 0)
        e -= p0 * log2f(p0);
    if (p1 > 0)
        e -= p1 * log2f(p1);
    return e;
}

// Kernel: evaluate one split per thread
__global__ void compute_impurity(
    const float *data, const int *labels, const int *indices,
    int n, int n_feats, int n_thr, const float *thr, float *imps)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_feats * n_thr;
    if (j >= total)
        return;

    int f = j / n_thr;
    float t = thr[j];

    int l0 = 0, l1 = 0, r0 = 0, r1 = 0;
    for (int k = 0; k < n; ++k)
    {
        int id = indices[k];
        float v = data[id * n_feats + f];
        int lab = labels[id];
        if (v <= t)
        {
            if (lab)
                ++l1;
            else
                ++l0;
        }
        else
        {
            if (lab)
                ++r1;
            else
                ++r0;
        }
    }

    float ln = l0 + l1, rn = r0 + r1, N = ln + rn;
    float Hleft = entropy(l0, l1);
    float Hright = entropy(r0, r1);
    imps[j] = (ln / N) * Hleft + (rn / N) * Hright;
}

// Host wrapper: alloc/copy/launch/copy back/find best/free
extern "C" void find_best_split_cuda(
    const float *h_data, const int *h_labels, const int *h_idx,
    int n, int n_feats, int n_thr,
    const float *h_thr, int *out_f, float *out_t)
{
    int total = n_feats * n_thr;
    size_t dsam = n * n_feats * sizeof(float);
    size_t dint = n * sizeof(int);
    size_t dthr = total * sizeof(float);
    size_t dimp = total * sizeof(float);

    float *d_data, *d_thr, *d_imp;
    int *d_lbl, *d_idx;
    cudaMalloc(&d_data, dsam);
    cudaMalloc(&d_lbl, dint);
    cudaMalloc(&d_idx, dint);
    cudaMalloc(&d_thr, dthr);
    cudaMalloc(&d_imp, dimp);

    cudaMemcpy(d_data, h_data, dsam, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lbl, h_labels, dint, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, dint, cudaMemcpyHostToDevice);
    cudaMemcpy(d_thr, h_thr, dthr, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    compute_impurity<<<blocks, threads>>>(
        d_data, d_lbl, d_idx, n, n_feats, n_thr, d_thr, d_imp);
    cudaDeviceSynchronize();

    float *h_imp = (float *)malloc(dimp);
    cudaMemcpy(h_imp, d_imp, dimp, cudaMemcpyDeviceToHost);

    float best = FLT_MAX;
    int bi = 0;
    for (int i = 0; i < total; ++i)
    {
        if (h_imp[i] < best)
        {
            best = h_imp[i];
            bi = i;
        }
    }
    *out_f = bi / n_thr;
    *out_t = h_thr[bi];

    free(h_imp);
    cudaFree(d_data);
    cudaFree(d_lbl);
    cudaFree(d_idx);
    cudaFree(d_thr);
    cudaFree(d_imp);
}
