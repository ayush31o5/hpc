#include <stdio.h>
#include <stdlib.h>
#include "./include/decision_tree.h"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s data/student-data.csv\n", argv[0]);
        return EXIT_FAILURE;
    }

    // 1) Load data
    float data[MAX_SAMPLES][MAX_FEATURES];
    int labels[MAX_SAMPLES];
    int n = load_csv(argv[1], data, labels);
    if (n <= 0)
    {
        fprintf(stderr, "Failed to load data or no samples found\n");
        return EXIT_FAILURE;
    }

    // 2) Split into 80/20 train/test
    int train_n = (int)(0.8 * n), test_n = n - train_n;
    int *train_idx = malloc(train_n * sizeof(int));
    int *test_idx = malloc(test_n * sizeof(int));
    for (int i = 0; i < train_n; ++i)
        train_idx[i] = i;
    for (int i = 0; i < test_n; ++i)
        test_idx[i] = train_n + i;

    // 3) Train tree (depth=4)
    TreeNode *root = train_tree(train_idx, train_n, 4, data, labels);

    // 4) Evaluate on test set
    int correct = 0;
    for (int i = 0; i < test_n; ++i)
    {
        int idx = test_idx[i];
        int pred = predict_tree(root, data[idx]);
        if (pred == labels[idx])
            correct++;
    }
    printf("Test accuracy: %.2f%% (%d/%d)\n",
           100.0f * correct / test_n, correct, test_n);

    // 5) Cleanup
    free_tree(root);
    free(train_idx);
    free(test_idx);
    return EXIT_SUCCESS;
}
