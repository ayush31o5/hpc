#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "decision_tree.h"

// which CSV columns map to which features (0-based):
// see header comments in decision_tree.h for column indices

static int map_str(const char *v, const char *choices[], int m)
{
    for (int i = 0; i < m; ++i)
        if (strcmp(v, choices[i]) == 0)
            return i;
    return 0;
}

// 1) Load & encode all features & label
int load_csv(const char *path, float data[][MAX_FEATURES], int labels[])
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char line[2048];
    if (!fgets(line, sizeof(line), f))
    {
        fclose(f);
        return -1;
    }

    const char *mjob_vals[] = {"at_home", "health", "other", "services", "teacher"};
    const char *fjob_vals[] = {"at_home", "health", "other", "services", "teacher"};
    const char *reason_vals[] = {"course", "home", "other", "reputation"};
    const char *guard_vals[] = {"mother", "father", "other"};

    int n = 0;
    while (n < MAX_SAMPLES && fgets(line, sizeof(line), f))
    {
        char *cols[31];
        char *save = NULL;
        cols[0] = strtok_r(line, ",", &save);
        for (int i = 1; i < 31; ++i)
            cols[i] = strtok_r(NULL, ",", &save);

        int fi = 0;
        // numeric features
        data[n][fi++] = atof(cols[2]);
        data[n][fi++] = atof(cols[6]);
        data[n][fi++] = atof(cols[7]);
        data[n][fi++] = atof(cols[12]);
        data[n][fi++] = atof(cols[13]);
        data[n][fi++] = atof(cols[14]);
        data[n][fi++] = atof(cols[23]);
        data[n][fi++] = atof(cols[24]);
        data[n][fi++] = atof(cols[25]);
        data[n][fi++] = atof(cols[26]);
        data[n][fi++] = atof(cols[27]);
        data[n][fi++] = atof(cols[29]);
        // binary categoricals
        data[n][fi++] = (cols[1][0] == 'M');
        data[n][fi++] = (cols[3][0] == 'U');
        data[n][fi++] = (strcmp(cols[4], "GT3") == 0);
        data[n][fi++] = (cols[5][0] == 'T');
        data[n][fi++] = (cols[15][0] == 'y');
        data[n][fi++] = (cols[16][0] == 'y');
        data[n][fi++] = (cols[17][0] == 'y');
        data[n][fi++] = (cols[18][0] == 'y');
        data[n][fi++] = (cols[19][0] == 'y');
        data[n][fi++] = (cols[20][0] == 'y');
        data[n][fi++] = (cols[21][0] == 'y');
        data[n][fi++] = (cols[22][0] == 'y');
        // multi‐category
        data[n][fi++] = map_str(cols[8], mjob_vals, 5);
        data[n][fi++] = map_str(cols[9], fjob_vals, 5);
        data[n][fi++] = map_str(cols[10], reason_vals, 4);
        data[n][fi++] = map_str(cols[11], guard_vals, 3);
        // leave indices 28–30 for main.c interactions
        labels[n] = (cols[30][0] == 'y') ? 1 : 0;
        ++n;
    }
    fclose(f);
    return n;
}

// majority vote
static int majority_label(int *idx, int n, int *labels)
{
    int c0 = 0, c1 = 0;
    for (int i = 0; i < n; ++i)
        if (labels[idx[i]])
            ++c1;
        else
            ++c0;
    return (c1 >= c0);
}

// float comparator
static int float_cmp(const void *a, const void *b)
{
    float fa = *(const float *)a, fb = *(const float *)b;
    return fa < fb ? -1 : fa > fb ? 1
                                  : 0;
}

// build quantile thresholds
static void build_thresholds(int *idx, int n,
                             float data[][MAX_FEATURES], float *thr, int n_feats, int n_thr)
{
    float *vals = malloc(n * sizeof(float));
    for (int f = 0; f < n_feats; ++f)
    {
        for (int i = 0; i < n; ++i)
            vals[i] = data[idx[i]][f];
        qsort(vals, n, sizeof(float), float_cmp);
        for (int t = 0; t < n_thr; ++t)
        {
            int j = (t + 1) * n / (n_thr + 1);
            if (j >= n - 1)
                j = n - 2;
            thr[f * n_thr + t] = 0.5f * (vals[j] + vals[j + 1]);
        }
    }
    free(vals);
}

// recursive trainer with regularization
TreeNode *train_tree(int *idx, int n, int depth,
                     float data[][MAX_FEATURES], int labels[])
{
    TreeNode *node = malloc(sizeof(TreeNode));
    node->left = node->right = NULL;

    // stop if too shallow or too few samples
    if (depth == 0 || n < MIN_SAMPLES_SPLIT)
    {
        node->is_leaf = 1;
        node->prediction = majority_label(idx, n, labels);
        return node;
    }

    // build thresholds
    float *thr = malloc(MAX_FEATURES * MAX_THRESHOLDS * sizeof(float));
    build_thresholds(idx, n, data, thr, MAX_FEATURES, MAX_THRESHOLDS);

    // find best split (GPU)
    int bf;
    float bt;
    find_best_split_cuda(&data[0][0], labels, idx, n,
                         MAX_FEATURES, MAX_THRESHOLDS,
                         thr, &bf, &bt);
    free(thr);

    node->feature = bf;
    node->threshold = bt;
    node->is_leaf = 0;

    // partition
    int *lidx = malloc(n * sizeof(int)), *ridx = malloc(n * sizeof(int));
    int ln = 0, rn = 0;
    for (int i = 0; i < n; ++i)
    {
        int id = idx[i];
        if (data[id][bf] <= bt)
            lidx[ln++] = id;
        else
            ridx[rn++] = id;
    }

    // enforce minimum leaf size
    if (ln < MIN_SAMPLES_LEAF || rn < MIN_SAMPLES_LEAF)
    {
        node->is_leaf = 1;
        node->prediction = majority_label(idx, n, labels);
        free(lidx);
        free(ridx);
        return node;
    }

    // recurse
    node->left = train_tree(lidx, ln, depth - 1, data, labels);
    node->right = train_tree(ridx, rn, depth - 1, data, labels);
    free(lidx);
    free(ridx);
    return node;
}

// predict
int predict_tree(const TreeNode *node, const float sample[])
{
    if (node->is_leaf)
        return node->prediction;
    if (sample[node->feature] <= node->threshold)
        return predict_tree(node->left, sample);
    else
        return predict_tree(node->right, sample);
}

// free
void free_tree(TreeNode *node)
{
    if (!node)
        return;
    free_tree(node->left);
    free_tree(node->right);
    free(node);
}
