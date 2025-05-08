#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include "decision_tree.h"

// simple Fisher–Yates shuffle
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
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s data/student-data.csv\n", argv[0]);
        return EXIT_FAILURE;
    }
    srand(1234);

    static float data[MAX_SAMPLES][MAX_FEATURES];
    static int labels[MAX_SAMPLES];

    // 1) Load & encode features
    int n = load_csv(argv[1], data, labels);
    if (n <= 0)
    {
        fprintf(stderr, "Error loading data\n");
        return EXIT_FAILURE;
    }

    // 2) Add 3 interaction features
    for (int i = 0; i < n; ++i)
    {
        data[i][28] = data[i][4] * data[i][5];  // studytime * failures
        data[i][29] = data[i][11] * data[i][9]; // absences * Dalc
        data[i][30] = data[i][7] * data[i][8];  // freetime * goout
    }

    // 3) Normalize each feature to [0,1]
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

    // 5) Train tree (depth=12)
    TreeNode *root = train_tree(train_idx, train_n, 12, data, labels);

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
