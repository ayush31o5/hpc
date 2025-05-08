#include <cuda_runtime.h>
#include <float.h>
#include "decision_tree.h"

// CUDA kernel: evaluate impurity for one candidate split
__global__ void compute_impurity(
    const float *data, const int *labels, const int *indices,
    int n, int n_feats, int n_thr,
    const float *thr, float *imps)
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
            (lab ? l1 : l0)++;
        else
            (lab ? r1 : r0)++;
    }
    float ln = l0 + l1, rn = r0 + r1, N = ln + rn;
    float gL = ln > 0 ? 1 - (l0 / ln) * (l0 / ln) - (l1 / ln) * (l1 / ln) : 0;
    float gR = rn > 0 ? 1 - (r0 / rn) * (r0 / rn) - (r1 / rn) * (r1 / rn) : 0;
    imps[j] = (ln / N) * gL + (rn / N) * gR;
}

// Host wrapper
extern "C" void find_best_split_cuda(
    const float *h_data, const int *h_labels, const int *h_indices,
    int n, int n_feats, int n_thr,
    const float *h_thr, int *out_f, float *out_t)
{
    int total = n_feats * n_thr;
    size_t dsam = n * n_feats * sizeof(float);
    size_t ds = n * sizeof(int);
    size_t dth = total * sizeof(float);
    size_t dimp = total * sizeof(float);

    float *d_data, *d_thr, *d_imp;
    int *d_lbl, *d_idx;
    cudaMalloc(&d_data, dsam);
    cudaMalloc(&d_lbl, ds);
    cudaMalloc(&d_idx, ds);
    cudaMalloc(&d_thr, dth);
    cudaMalloc(&d_imp, dimp);

    cudaMemcpy(d_data, h_data, dsam, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lbl, h_labels, ds, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_indices, ds, cudaMemcpyHostToDevice);
    cudaMemcpy(d_thr, h_thr, dth, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    compute_impurity<<<blocks, threads>>>(
        d_data, d_lbl, d_idx, n, n_feats, n_thr, d_thr, d_imp);
    cudaDeviceSynchronize();

    float *h_imp = (float *)malloc(dimp);
    cudaMemcpy(h_imp, d_imp, dimp, cudaMemcpyDeviceToHost);

    // find best
    float best = FLT_MAX;
    int bi = 0;
    for (int i = 0; i < total; ++i)
        if (h_imp[i] < best)
        {
            best = h_imp[i];
            bi = i;
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
