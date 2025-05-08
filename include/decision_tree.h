#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <float.h>

#define MAX_FEATURES 31 // 28 base/categorical + 3 interactions
#define MAX_SAMPLES 10000
#define MAX_THRESHOLDS 200 // split resolution

// Regularization
#define MIN_SAMPLES_SPLIT 10 // require at least this many samples to attempt a split
#define MIN_SAMPLES_LEAF 5   // require at least this many samples in each leaf

    typedef struct TreeNode
    {
        int feature;
        float threshold;
        int is_leaf;
        int prediction;
        struct TreeNode *left, *right;
    } TreeNode;

    // load all features into data[][] and labels[]
    int load_csv(const char *path, float data[][MAX_FEATURES], int labels[]);

    // train a CART tree up to max_depth
    TreeNode *train_tree(int *indices, int n, int max_depth,
                         float data[][MAX_FEATURES], int labels[]);

    // predict one sample
    int predict_tree(const TreeNode *node, const float sample[]);

    // free the tree
    void free_tree(TreeNode *node);

    // GPU-accelerated split finder
    void find_best_split_cuda(const float *data, const int *labels, const int *indices,
                              int n, int n_feats, int n_thr,
                              const float *thresholds,
                              int *out_feature, float *out_threshold);

#ifdef __cplusplus
}
#endif

#endif // DECISION_TREE_H
