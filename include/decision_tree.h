#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#define MAX_FEATURES 12
#define MAX_SAMPLES 10000 // adjust to your dataset size
#define MAX_THRESHOLDS 20 // thresholds per feature

// Tree node
typedef struct TreeNode
{
    int feature;     // split feature index
    float threshold; // split threshold
    int is_leaf;     // leaf flag
    int prediction;  // predicted label (0 or 1)
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

// Load CSV into data[][] and labels[], return sample count
int load_csv(const char *path, float data[][MAX_FEATURES], int labels[]);

// Train CART tree up to max_depth, return root
TreeNode *train_tree(int *indices, int n, int max_depth,
                     float data[][MAX_FEATURES], int labels[]);

// Predict single sample
int predict_tree(TreeNode *node, const float sample[]);

// Free the tree
void free_tree(TreeNode *node);

// GPU‐accelerated split finder
void find_best_split_cuda(const float *data, const int *labels, const int *indices,
                          int n, int n_feats, int n_thresh,
                          const float *thresholds,
                          int *out_feature, float *out_threshold);

#endif // DECISION_TREE_H
