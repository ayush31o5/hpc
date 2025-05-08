#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>         // clock_gettime
#include <cuda_runtime.h> // CUDA timing APIs
#include <float.h>
#include "decision_tree.h"

// Fisher–Yates shuffle
static void shuffle(int *a, int n)
{
    for (int i = n - 1; i > 0; --i)
    {
        int j = rand() % (i + 1);
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        fprintf(stderr, "Usage: %s data/student-data.csv [cpu|gpu]\n", argv[0]);
        return EXIT_FAILURE;
    }
    int use_gpu = 1;
    if (argc == 3 && strcmp(argv[2], "cpu") == 0)
        use_gpu = 0;

    srand(1234);

    static float data[MAX_SAMPLES][MAX_FEATURES];
    static int labels[MAX_SAMPLES];

    // 1) Load & encode
    int n = load_csv(argv[1], data, labels);
    if (n <= 0)
    {
        fprintf(stderr, "Error loading data\n");
        return EXIT_FAILURE;
    }

    // 2) Add 3 interaction features
    for (int i = 0; i < n; ++i)
    {
        data[i][28] = data[i][4] * data[i][5];
        data[i][29] = data[i][11] * data[i][9];
        data[i][30] = data[i][7] * data[i][8];
    }

    // 3) Normalize all features to [0,1]
    for (int f = 0; f < MAX_FEATURES; ++f)
    {
        float mn = FLT_MAX, mx = -FLT_MAX;
        for (int i = 0; i < n; ++i)
        {
            float v = data[i][f];
            if (v < mn)
                mn = v;
            if (v > mx)
                mx = v;
        }
        float range = (mx - mn) > 1e-6f ? (mx - mn) : 1.0f;
        for (int i = 0; i < n; ++i)
        {
            data[i][f] = (data[i][f] - mn) / range;
        }
    }

    // 4) Shuffle & split 80/20
    int *idx = malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i)
        idx[i] = i;
    shuffle(idx, n);
    int train_n = (int)(0.8f * n), test_n = n - train_n;
    int *train_idx = idx;
    int *test_idx = idx + train_n;

    // 5) Train & time
    TreeNode *root;
    if (!use_gpu)
    {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        root = train_tree(train_idx, train_n, 12, data, labels);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double cpu_secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        printf("CPU training time: %.6f s\n", cpu_secs);
    }
    else
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // **Fixed: pass the stream argument (0)**
        cudaEventRecord(start, 0);
        root = train_tree(train_idx, train_n, 12, data, labels);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float gpu_ms;
        cudaEventElapsedTime(&gpu_ms, start, stop);
        printf("GPU training time: %.3f ms\n", gpu_ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // 6) Evaluate
    int correct = 0;
    for (int i = 0; i < test_n; ++i)
    {
        int s = test_idx[i];
        if (predict_tree(root, data[s]) == labels[s])
            ++correct;
    }
    printf("Test accuracy: %.2f%% (%d/%d)\n",
           100.0f * correct / test_n, correct, test_n);

    // 7) Cleanup
    free_tree(root);
    free(idx);
    return EXIT_SUCCESS;
}
