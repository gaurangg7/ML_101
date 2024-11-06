#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

// Structure for tree nodes
struct Node {
    double value;      // For leaf nodes, the prediction value
    int feature;       // Feature index to split on
    double threshold;  // Threshold value for the split
    Node* left;        // Left child
    Node* right;       // Right child

    Node(double val) : value(val), feature(-1), threshold(0.0), left(nullptr), right(nullptr) {}
};

// Structure to hold dataset
struct Dataset {
    vector<vector<double>> X;
    vector<double> y;
};

// Structure to hold split datasets
struct SplitDataset {
    Dataset train;
    Dataset validation;
};

// Function to split dataset into training and validation sets
SplitDataset splitData(const vector<vector<double>>& X, const vector<double>& y, double validation_ratio=0.2) {
    size_t total = X.size();
    size_t validation_size = static_cast<size_t>(total * validation_ratio);
    
    vector<size_t> indices(total);
    for (size_t i = 0; i < total; ++i) indices[i] = i;
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);
    
    Dataset train, validation;
    for (size_t i = 0; i < total; ++i) {
        if (i < validation_size) {
            validation.X.push_back(X[indices[i]]);
            validation.y.push_back(y[indices[i]]);
        } else {
            train.X.push_back(X[indices[i]]);
            train.y.push_back(y[indices[i]]);
        }
    }
    
    return SplitDataset{train, validation};
}

// Function to calculate Mean Squared Error (MSE)
double calculateMSE(const vector<double>& y) {
    if (y.empty()) return 0.0;
    double mean = 0.0;
    for (double val : y) mean += val;
    mean /= y.size();

    double mse = 0.0;
    for (double val : y) mse += pow(val - mean, 2);
    return mse / y.size();
}

// Function to split the dataset based on feature and threshold
void splitDataset(const vector<vector<double>>& X, const vector<double>& y, int feature, double threshold, 
                 vector<vector<double>>& leftX, vector<double>& leftY, 
                 vector<vector<double>>& rightX, vector<double>& rightY) {
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][feature] <= threshold) {
            leftX.push_back(X[i]);
            leftY.push_back(y[i]);
        } else {
            rightX.push_back(X[i]);
            rightY.push_back(y[i]);
        }
    }
}

// Function to find the best split (feature and threshold) that minimizes MSE
pair<int, double> findBestSplit(const vector<vector<double>>& X, const vector<double>& y) {
    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestMSE = numeric_limits<double>::max();

    if (X.empty()) return {bestFeature, bestThreshold};
    size_t numFeatures = X[0].size();

    for (size_t i = 0; i < numFeatures; ++i) {
        vector<double> featureValues;
        for (const auto& row : X) featureValues.push_back(row[i]);
        sort(featureValues.begin(), featureValues.end());
        featureValues.erase(unique(featureValues.begin(), featureValues.end(), 
            [&](double a, double b) { return abs(a - b) < 1e-6; }), featureValues.end());

        for (size_t j = 1; j < featureValues.size(); ++j) {
            double threshold = (featureValues[j - 1] + featureValues[j]) / 2.0;

            vector<vector<double>> leftX, rightX;
            vector<double> leftY, rightY;

            splitDataset(X, y, i, threshold, leftX, leftY, rightX, rightY);

            if (leftY.empty() || rightY.empty()) continue;

            double mse_left = calculateMSE(leftY);
            double mse_right = calculateMSE(rightY);
            double weighted_mse = (mse_left * leftY.size() + mse_right * rightY.size()) / y.size();

            if (weighted_mse < bestMSE) {
                bestFeature = i;
                bestThreshold = threshold;
                bestMSE = weighted_mse;
            }
        }
    }

    return {bestFeature, bestThreshold};
}

// Function to build the decision tree recursively
Node* buildTree(const vector<vector<double>>& X, const vector<double>& y, int depth = 0, int maxDepth = 5) {
    double mean = 0.0;
    for (double val : y) mean += val;
    mean /= y.size();
    Node* node = new Node(mean);

    if (depth >= maxDepth || y.size() <= 1) return node;

    auto [bestFeature, bestThreshold] = findBestSplit(X, y);
    if (bestFeature == -1) return node;

    vector<vector<double>> leftX, rightX;
    vector<double> leftY, rightY;
    splitDataset(X, y, bestFeature, bestThreshold, leftX, leftY, rightX, rightY);

    if (leftY.empty() || rightY.empty()) return node;

    node->feature = bestFeature;
    node->threshold = bestThreshold;
    node->left = buildTree(leftX, leftY, depth + 1, maxDepth);
    node->right = buildTree(rightX, rightY, depth + 1, maxDepth);

    return node;
}

// Function to make a prediction for a single sample
double predict(Node* node, const vector<double>& x) {
    if (!node->left && !node->right) return node->value;

    if (x[node->feature] <= node->threshold) return predict(node->left, x);
    else return predict(node->right, x);
}

// Function to compute MSE between predictions and actual values
double computeMSE(const vector<double>& predictions, const vector<double>& actual) {
    if (predictions.empty()) return 0.0;
    double mse = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double diff = predictions[i] - actual[i];
        mse += diff * diff;
    }
    return mse / predictions.size();
}

// Function to traverse the tree and collect predictions
void collectPredictions(Node* node, const vector<vector<double>>& X, vector<double>& preds) {
    for (const auto& x : X) preds.push_back(predict(node, x));
}

// Function to prune the tree using the validation set
double pruneTree(Node* node, const vector<vector<double>>& X_val, const vector<double>& y_val) {
    if (!node->left && !node->right) {
        double mse = 0.0;
        for (double val : y_val) mse += pow(val - node->value, 2);
        return mse / y_val.size();
    }
    
    vector<vector<double>> leftX, rightX;
    vector<double> leftY, rightY;
    for (size_t i = 0; i < X_val.size(); ++i) {
        if (X_val[i][node->feature] <= node->threshold) {
            leftX.push_back(X_val[i]);
            leftY.push_back(y_val[i]);
        } else {
            rightX.push_back(X_val[i]);
            rightY.push_back(y_val[i]);
        }
    }
    
    double mse_left = 0.0, mse_right = 0.0;
    if (node->left) mse_left = pruneTree(node->left, leftX, leftY);
    if (node->right) mse_right = pruneTree(node->right, rightX, rightY);
    
    vector<double> preds_before;
    collectPredictions(node, X_val, preds_before);
    double mse_before = computeMSE(preds_before, y_val);
    
    double mean = 0.0;
    for (double val : y_val) mean += val;
    mean /= y_val.size();
    
    double mse_pruned = 0.0;
    for (double val : y_val) mse_pruned += pow(val - mean, 2);
    mse_pruned /= y_val.size();
    
    if (mse_pruned <= mse_before) {
        delete node->left;
        delete node->right;
        node->left = nullptr;
        node->right = nullptr;
        node->value = mean;
        return mse_pruned;
    } else {
        return mse_before;
    }
}

// Function to delete the tree and free memory
void deleteTree(Node* node) {
    if (!node) return;
    deleteTree(node->left);
    deleteTree(node->right);
    delete node;
}

int main() {
    vector<vector<double>> X = {
        {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}, {10.0}, {11.0}
    };
    vector<double> y = {3.0, 6.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0};
    
    // Split the data into training and validation sets
    SplitDataset split = splitData(X, y, 0.3); // 70% training, 30% validation
    
    // Build the tree using the training data
    int maxDepth = 5;
    Node* tree = buildTree(split.train.X, split.train.y, 0, maxDepth);
    
    // Prune the tree using the validation data
    pruneTree(tree, split.validation.X, split.validation.y);
    
    // Predicting for a new data point
    vector<double> test = {4.5};
    cout << "Prediction for 4.5: " << predict(tree, test) << endl;
    
    // Evaluate the tree on validation set
    vector<double> val_preds;
    for (const auto& x : split.validation.X) {
        val_preds.push_back(predict(tree, x));
    }
    double val_mse = computeMSE(val_preds, split.validation.y);
    cout << "Validation MSE after pruning: " << val_mse << endl;
    
    // Clean up the tree (memory management)
    deleteTree(tree);
    
    return 0;
}