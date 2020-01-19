//
// Created by vloods on 1/4/20.
//

#ifndef RPFOREST_RPTREE_H
#define RPFOREST_RPTREE_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/SparseCore"

class RPForest{
public:
    RPForest(const Eigen::Ref<const Eigen::MatrixXf> &X_) :
            X(Eigen::Map<const Eigen::MatrixXf>(X_.data(), X_.rows(), X_.cols())),
            n_samples(X_.cols()),
            dim(X_.rows()) {}

    RPForest(const float *X_, int dim_, int n_samples_) :
            X(Eigen::Map<const Eigen::MatrixXf>(X_, dim_, n_samples_)),
            n_samples(n_samples_),
            dim(dim_) {}

    void grow(int n_trees_, int depth_, float density_ = -1.0, int seed = 0){
        if(!is_empty()){
            throw std::logic_error("Forest has already been grown.");
        }

        if (n_trees_ <= 0) {
            throw std::out_of_range("The number of trees must be positive.");
        }

        if (depth_ <= 0 || depth_ > std::log2(n_samples)) {
            throw std::out_of_range("The depth must belong to the set {1, ... , log2(n)}.");
        }

        if (density_ < -1.0001 || density_ > 1.0001 || (density_ > -0.9999 && density_ < -0.0001)) {
            throw std::out_of_range("The density must be on the interval (0,1].");
        }

        n_trees = n_trees_;
        depth = depth_;
        n_pool = n_trees_ * depth_;
        n_array = 1 << (depth_ + 1);

        if (density_ < 0) {
            density = 1.0 / std::sqrt(dim);
        } else {
            density = density_;
        }

        density < 1 ? build_sparse_random_matrix(sparse_random_matrix, n_pool, dim, density, seed) :
                      build_dense_random_matrix(dense_random_matrix, n_pool, dim, seed);

        split_points = Eigen::MatrixXf(n_array, n_trees);
        tree_leaves = std::vector<std::vector<int>>(n_trees);

        count_first_leaf_indices_all(leaf_first_indices_all, n_samples, depth);
        leaf_first_indices = leaf_first_indices_all[depth];

        #pragma omp parallel for
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            Eigen::MatrixXf tree_projections;

            if (density < 1)
                tree_projections.noalias() = sparse_random_matrix.middleRows(n_tree * depth, depth) * X;
            else
                tree_projections.noalias() = dense_random_matrix.middleRows(n_tree * depth, depth) * X;

            tree_leaves[n_tree] = std::vector<int>(n_samples);
            std::vector<int> &indices = tree_leaves[n_tree];
            std::iota(indices.begin(), indices.end(), 0);

            grow_subtree(indices.begin(), indices.end(), 0, 0, n_tree, tree_projections);
        }
    }

    void grow_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                      int tree_level, int i, int n_tree, const Eigen::MatrixXf &tree_projections) {
        int n = end - begin;
        int idx_left = 2 * i + 1;
        int idx_right = idx_left + 1;

        if (tree_level == depth) return;

        std::nth_element(begin, begin + n / 2, end,
                         [&tree_projections, tree_level] (int i1, int i2) {
                             return tree_projections(tree_level, i1) < tree_projections(tree_level, i2);
                         });
        auto mid = end - n / 2;

        if (n % 2) {
            split_points(i, n_tree) = tree_projections(tree_level, *(mid - 1));
        } else {
            auto left_it = std::max_element(begin, mid,
                                            [&tree_projections, tree_level] (int i1, int i2) {
                                                return tree_projections(tree_level, i1) < tree_projections(tree_level, i2);
                                            });
            split_points(i, n_tree) = (tree_projections(tree_level, *mid) +
                                       tree_projections(tree_level, *left_it)) / 2.0;
        }

        grow_subtree(begin, mid, tree_level + 1, idx_left, n_tree, tree_projections);
        grow_subtree(mid, end, tree_level + 1, idx_right, n_tree, tree_projections);
    }

    void query(const float *data, int k, int vote_threshold, int *out,
               float *out_distances = nullptr, int *out_n_elected = nullptr) const {

        if (k <= 0 || k > n_samples) {
            throw std::out_of_range("k must belong to the set {1, ..., n}.");
        }

        if (vote_threshold <= 0 || vote_threshold > n_trees) {
            throw std::out_of_range("vote_threshold must belong to the set {1, ... , n_trees}.");
        }

        if (is_empty()) {
            throw std::logic_error("The index must be built before making queries.");
        }

        const Eigen::Map<const Eigen::VectorXf> q(data, dim);

        Eigen::VectorXf projected_query(n_pool);
        if (density < 1)
            projected_query.noalias() = sparse_random_matrix * q;
        else
            projected_query.noalias() = dense_random_matrix * q;

        std::vector<int> found_leaves(n_trees);

        /*
        * The following loops over all trees, and routes the query to exactly one
        * leaf in each.
        */
        #pragma omp parallel for
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            int idx_tree = 0;
            for (int d = 0; d < depth; ++d) {
                const int j = n_tree * depth + d;
                const int idx_left = 2 * idx_tree + 1;
                const int idx_right = idx_left + 1;
                const float split_point = split_points(idx_tree, n_tree);
                if (projected_query(j) <= split_point) {
                    idx_tree = idx_left;
                } else {
                    idx_tree = idx_right;
                }
            }
            found_leaves[n_tree] = idx_tree - (1 << depth) + 1;
        }

        int n_elected = 0, max_leaf_size = n_samples / (1 << depth) + 1;
        Eigen::VectorXi elected(n_trees * max_leaf_size);
        Eigen::VectorXi votes = Eigen::VectorXi::Zero(n_samples);

        // count votes
        for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
            int leaf_begin = leaf_first_indices[found_leaves[n_tree]];
            int leaf_end = leaf_first_indices[found_leaves[n_tree] + 1];
            const std::vector<int> &indices = tree_leaves[n_tree];
            for (int i = leaf_begin; i < leaf_end; ++i) {
                int idx = indices[i];
                if (++votes(idx) == vote_threshold)
                    elected(n_elected++) = idx;
            }
        }

        if (out_n_elected) {
            *out_n_elected = n_elected;
        }

        exact_knn(q, k, elected, n_elected, out, out_distances);
    }

    void query(const Eigen::Ref<const Eigen::VectorXf> &q, int k, int vote_threshold, int *out,
               float *out_distances = nullptr, int *out_n_elected = nullptr) const {
        query(q.data(), k, vote_threshold, out, out_distances, out_n_elected);
    }

    static void exact_knn(const float *q_data, const float *X_data, int dim, int n_samples,
                          int k, int *out, float *out_distances = nullptr) {

        const Eigen::Map<const Eigen::MatrixXf> X(X_data, dim, n_samples);
        const Eigen::Map<const Eigen::VectorXf> q(q_data, dim);

        if (k < 1 || k > n_samples) {
            throw std::out_of_range("k must be positive and no greater than the sample size of data X.");
        }

        Eigen::VectorXf distances(n_samples);

        #pragma omp parallel for
        for (int i = 0; i < n_samples; ++i)
            distances(i) = (X.col(i) - q).squaredNorm();

        if (k == 1) {
            Eigen::MatrixXf::Index index;
            distances.minCoeff(&index);
            out[0] = index;

            if (out_distances)
                out_distances[0] = std::sqrt(distances(index));

            return;
        }

        Eigen::VectorXi idx(n_samples);
        std::iota(idx.data(), idx.data() + n_samples, 0);
        std::partial_sort(idx.data(), idx.data() + k, idx.data() + n_samples,
                          [&distances](int i1, int i2) { return distances(i1) < distances(i2); });

        for (int i = 0; i < k; ++i)
            out[i] = idx(i);

        if (out_distances) {
            for (int i = 0; i < k; ++i)
                out_distances[i] = std::sqrt(distances(idx(i)));
        }
    }

    void exact_knn(const Eigen::Map<const Eigen::VectorXf> &q, int k, const Eigen::VectorXi &indices,
                   int n_elected, int *out, float *out_distances = nullptr) const {

        if (!n_elected) {
            for (int i = 0; i < k; ++i)
                out[i] = -1;

            if (out_distances) {
                for (int i = 0; i < k; ++i)
                    out_distances[i] = -1;
            }

            return;
        }

        Eigen::VectorXf distances(n_elected);

        #pragma omp parallel for
        for (int i = 0; i < n_elected; ++i)
            distances(i) = (X.col(indices(i)) - q).squaredNorm();

        if (k == 1) {
            Eigen::MatrixXf::Index index;
            distances.minCoeff(&index);
            out[0] = n_elected ? indices(index) : -1;

            if (out_distances)
                out_distances[0] = n_elected ? std::sqrt(distances(index)) : -1;

            return;
        }

        int n_to_sort = n_elected > k ? k : n_elected;
        Eigen::VectorXi idx(n_elected);
        std::iota(idx.data(), idx.data() + n_elected, 0);
        std::partial_sort(idx.data(), idx.data() + n_to_sort, idx.data() + n_elected,
                          [&distances](int i1, int i2) { return distances(i1) < distances(i2); });

        for (int i = 0; i < k; ++i)
            out[i] = i < n_elected ? indices(idx(i)) : -1;

        if (out_distances) {
            for (int i = 0; i < k; ++i)
                out_distances[i] = i < n_elected ? std::sqrt(distances(idx(i))) : -1;
        }
    }

    static void exact_knn(const Eigen::Ref<const Eigen::VectorXf> &q,
                          const Eigen::Ref<const Eigen::MatrixXf> &X,
                          int k, int *out, float *out_distances = nullptr) {
        RPForest::exact_knn(q.data(), X.data(), X.rows(), X.cols(), k, out, out_distances);
    }

    void exact_knn(const float *q, int k, int *out, float *out_distances = nullptr) const {
        RPForest::exact_knn(q, X.data(), dim, n_samples, k, out, out_distances);
    }

    void exact_knn(const Eigen::Ref<const Eigen::VectorXf> &q, int k, int *out,
                   float *out_distances = nullptr) const {
        RPForest::exact_knn(q.data(), X.data(), dim, n_samples, k, out, out_distances);
    }

    static void count_leaf_sizes(int n, int level, int tree_depth, std::vector<int> &out_leaf_sizes) {
        if (level == tree_depth) {
            out_leaf_sizes.push_back(n);
            return;
        }

        count_leaf_sizes(n - n / 2, level + 1, tree_depth, out_leaf_sizes);
        count_leaf_sizes(n / 2, level + 1, tree_depth, out_leaf_sizes);
    }

    static void count_first_leaf_indices(std::vector<int> &indices, int n, int depth) {
        std::vector<int> leaf_sizes;
        count_leaf_sizes(n, 0, depth, leaf_sizes);

        indices = std::vector<int>(leaf_sizes.size() + 1);
        indices[0] = 0;
        for (int i = 0; i < (int) leaf_sizes.size(); ++i)
            indices[i + 1] = indices[i] + leaf_sizes[i];
    }

    static void count_first_leaf_indices_all(std::vector<std::vector<int>> &indices, int n, int depth_max) {
        for (int d = 0; d <= depth_max; ++d) {
            std::vector<int> idx;
            count_first_leaf_indices(idx, n, d);
            indices.push_back(idx);
        }
    }

    static void build_sparse_random_matrix(Eigen::SparseMatrix<float, Eigen::RowMajor> &sparse_random_matrix,
                                           int n_row, int n_col, float density, int seed = 0) {
        sparse_random_matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>(n_row, n_col);

        std::random_device rd;
        int s = seed ? seed : rd();
        std::mt19937 gen(s);
        std::uniform_real_distribution<float> uni_dist(0, 1);
        std::normal_distribution<float> norm_dist(0, 1);

        std::vector<Eigen::Triplet<float>> triplets;
        for (int j = 0; j < n_row; ++j) {
            for (int i = 0; i < n_col; ++i) {
                if (uni_dist(gen) > density) continue;
                triplets.push_back(Eigen::Triplet<float>(j, i, norm_dist(gen)));
            }
        }

        sparse_random_matrix.setFromTriplets(triplets.begin(), triplets.end());
        sparse_random_matrix.makeCompressed();
    }

    static void build_dense_random_matrix(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &dense_random_matrix,
                                          int n_row, int n_col, int seed = 0) {
        dense_random_matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(n_row, n_col);

        std::random_device rd;
        int s = seed ? seed : rd();
        std::mt19937 gen(s);
        std::normal_distribution<float> normal_dist(0, 1);

        std::generate(dense_random_matrix.data(), dense_random_matrix.data() + n_row * n_col,
                      [&normal_dist, &gen] { return normal_dist(gen); });
    }

    bool is_empty() const {
        return n_trees == 0;
    }

private:
    const Eigen::Map<const Eigen::MatrixXf> X; // the data matrix
    Eigen::MatrixXf split_points; // all split points in all trees
    std::vector<std::vector<int>> tree_leaves; // contains all leaves of all trees
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dense_random_matrix; // random vectors needed for all the RP-trees
    Eigen::SparseMatrix<float, Eigen::RowMajor> sparse_random_matrix; // random vectors needed for all the RP-trees
    std::vector<std::vector<int>> leaf_first_indices_all; // first indices for each level
    std::vector<int> leaf_first_indices; // first indices of each leaf of tree in tree_leaves

    const int n_samples; // sample size of data
    const int dim; // dimension of data
    int n_trees{0}; // number of RP-trees
    int depth {0}; // depth of an RP-tree with median split
    float density{-1.0}; // expected ratio of non-zero components in a projection matrix
    int n_pool{0}; // amount of random vectors needed for all the RP-trees
    int n_array{0}; // length of the one RP-tree as array

};


#endif //RPFOREST_RPTREE_H
