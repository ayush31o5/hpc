#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "decision_tree.h"

// Choose columns from CSV (0-based)
static const int feat_cols[MAX_FEATURES] = {
    2, 6, 7, 12, 13, 14, 23, 24, 25, 26, 27, 29};

// Load CSV
int load_csv(const char *path, float data[][MAX_FEATURES], int labels[])
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char line[1024];
    fgets(line, sizeof(line), f); // skip header

    int n = 0;
    while (fgets(line, sizeof(line), f) && n < MAX_SAMPLES)
    {
        char *tok = strtok(line, ",");
        int fi = 0;
        for (int col = 0; tok && col <= feat_cols[MAX_FEATURES - 1]; ++col)
        {
            if (col == feat_cols[fi])
            {
                data[n][fi++] = atof(tok);
                if (fi == MAX_FEATURES)
                {
                    // read remaining columns up to label
                    while (col++ < feat_cols[MAX_FEATURES - 1])
                        tok = strtok(NULL, ",");
                    tok = strtok(NULL, ",");
                    labels[n] = (tok[0] == 'y' || tok[0] == 'Y') ? 1 : 0;
                    break;
                }
            }
            tok = strtok(NULL, ",");
        }
        n++;
    }
    fclose(f);
    return n;
}

// Majority vote for leaf
static int majority_label(int *idx, int n, int *labels)
{
    int c0 = 0, c1 = 0;
    for (int i = 0; i < n; ++i)
        (labels[idx[i]] ? c1 : c0)++;
    return (c1 >= c0);
}

// Train recursively
TreeNode *train_tree(int *indices, int n, int depth,
                     float data[][MAX_FEATURES], int labels[])
{
    TreeNode *node = malloc(sizeof(TreeNode));
    node->left = node->right = NULL;
    node->is_leaf = 0;

    // Stopping criteria
    if (depth == 0 || n < 5)
    {
        node->is_leaf = 1;
        node->prediction = majority_label(indices, n, labels);
        return node;
    }

    // Generate candidate thresholds
    float thr[MAX_FEATURES * MAX_THRESHOLDS];
    for (int f = 0; f < MAX_FEATURES; ++f)
    {
        float minv = FLT_MAX, maxv = -FLT_MAX;
        for (int i = 0; i < n; ++i)
        {
            float v = data[indices[i]][f];
            if (v < minv)
                minv = v;
            if (v > maxv)
                maxv = v;
        }
        for (int t = 0; t < MAX_THRESHOLDS; ++t)
            thr[f * MAX_THRESHOLDS + t] =
                minv + (t + 1) * (maxv - minv) / (MAX_THRESHOLDS + 1);
    }

    // Find best split on GPU
    int best_f;
    float best_th;
    find_best_split_cuda(&data[0][0], labels, indices, n,
                         MAX_FEATURES, MAX_THRESHOLDS,
                         thr, &best_f, &best_th);

    node->feature = best_f;
    node->threshold = best_th;

    // Partition indices
    int *lidx = malloc(n * sizeof(int)), *ridx = malloc(n * sizeof(int));
    int ln = 0, rn = 0;
    for (int i = 0; i < n; ++i)
    {
        int id = indices[i];
        if (data[id][best_f] <= best_th)
            lidx[ln++] = id;
        else
            ridx[rn++] = id;
    }

    // If no split gain, make leaf
    if (ln == 0 || rn == 0)
    {
        node->is_leaf = 1;
        node->prediction = majority_label(indices, n, labels);
        free(lidx);
        free(ridx);
        return node;
    }

    // Recurse
    node->left = train_tree(lidx, ln, depth - 1, data, labels);
    node->right = train_tree(ridx, rn, depth - 1, data, labels);
    free(lidx);
    free(ridx);
    return node;
}

// Predict one sample
int predict_tree(TreeNode *node, const float sample[])
{
    if (node->is_leaf)
        return node->prediction;
    if (sample[node->feature] <= node->threshold)
        return predict_tree(node->left, sample);
    else
        return predict_tree(node->right, sample);
}

// Free memory
void free_tree(TreeNode *node)
{
    if (!node)
        return;
    free_tree(node->left);
    free_tree(node->right);
    free(node);
}
