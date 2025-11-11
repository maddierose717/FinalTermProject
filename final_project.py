"""
CS 634 Data Mining - Final Term Project
Option 1: Supervised Data Mining (Classification)
Student Name: Madison Rose Lucas
NJIT UCID: MRL58
Email: MRL58@njit.edu
Algorithm 1: LIBSVM with RBF Kernel (Category 1: Support Vector Machines)
Algorithm 2: Random Forest (Category 2: Random Forests)

Implementation: From Scratch (using only NumPy for basic operations)
"""
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Algorithm 1: SVM with RBF Kernel (From Scratch)
# ============================================================================

class SVMFromScratch:
    """
    Support Vector Machine with RBF kernel implemented from scratch.
    Uses Sequential Minimal Optimization (SMO) algorithm for training.
    """
    def __init__(self, C=1.0, gamma='scale', max_iter=1000, tol=1e-3):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.alphas = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        
    def _rbf_kernel(self, X1, X2):
        """Compute RBF (Gaussian) kernel between X1 and X2."""
        if self.gamma == 'scale':
            gamma_val = 1.0 / (X1.shape[1] * X1.var())
        else:
            gamma_val = self.gamma
            
        # Compute squared Euclidean distances
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                   np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma_val * sq_dists)
    
    def _decision_function(self, X):
        """Compute decision function for samples in X."""
        kernel_matrix = self._rbf_kernel(X, self.X_train)
        return np.dot(kernel_matrix, self.alphas * self.y_train) + self.b
    
    def fit(self, X, y):
        """Train SVM using simplified SMO algorithm. For multiclass, uses One-vs-Rest strategy."""
        self.classes_ = np.unique(y)
        
        if len(self.classes_) == 2:
            y_binary = np.where(y == self.classes_[0], -1, 1)
            self._fit_binary(X, y_binary)
            self.binary = True
        else:
            self.binary = False
            self.classifiers = []
            for cls in self.classes_:
                y_binary = np.where(y == cls, 1, -1)
                clf = SVMFromScratch(C=self.C, gamma=self.gamma, 
                                    max_iter=self.max_iter, tol=self.tol)
                clf._fit_binary(X, y_binary)
                self.classifiers.append(clf)
        return self
    
    def _fit_binary(self, X, y):
        """Fit binary SVM using simplified SMO."""
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y
        self.alphas = np.zeros(n_samples)
        self.b = 0
        K = self._rbf_kernel(X, X)
        
        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alphas)
            
            for i in range(n_samples):
                Ei = self._decision_function(X[i].reshape(1, -1))[0] - y[i]
                
                if (y[i] * Ei < -self.tol and self.alphas[i] < self.C) or \
                   (y[i] * Ei > self.tol and self.alphas[i] > 0):
                    
                    j = np.random.choice([k for k in range(n_samples) if k != i])
                    Ej = self._decision_function(X[j].reshape(1, -1))[0] - y[j]
                    
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    self.alphas[j] -= y[j] * (Ei - Ej) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    b1 = self.b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (self.alphas[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
            
            if np.linalg.norm(self.alphas - alpha_prev) < self.tol:
                break
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        if self.binary:
            decision = self._decision_function(X)
            return np.where(decision >= 0, self.classes_[1], self.classes_[0])
        else:
            decisions = np.array([clf._decision_function(X) for clf in self.classifiers])
            return self.classes_[np.argmax(decisions, axis=0)]


# ============================================================================
# Algorithm 2: Random Forest (From Scratch)
# ============================================================================

class DecisionTreeFromScratch:
    """Decision Tree classifier implemented from scratch."""
    
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
        
    def _gini_impurity(self, y):
        """Calculate Gini impurity."""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _split_data(self, X, y, feature_idx, threshold):
        """Split data based on a feature and threshold."""
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])
    
    def _find_best_split(self, X, y, feature_indices):
        """Find the best feature and threshold to split on."""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                (X_left, y_left), (X_right, y_right) = self._split_data(X, y, feature_idx, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                n = len(y)
                gini = (len(y_left) / n) * self._gini_impurity(y_left) + \
                       (len(y_right) / n) * self._gini_impurity(y_right)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = np.bincount(y.astype(int)).argmax()
            return {'leaf': True, 'value': leaf_value}
        
        if self.max_features is not None:
            feature_indices = np.random.choice(n_features, 
                                             min(self.max_features, n_features), 
                                             replace=False)
        else:
            feature_indices = np.arange(n_features)
        
        best_feature, best_threshold = self._find_best_split(X, y, feature_indices)
        
        if best_feature is None:
            leaf_value = np.bincount(y.astype(int)).argmax()
            return {'leaf': True, 'value': leaf_value}
        
        (X_left, y_left), (X_right, y_right) = self._split_data(X, y, best_feature, best_threshold)
        
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """Build decision tree."""
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, tree):
        """Predict class for a single sample."""
        if tree['leaf']:
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForestFromScratch:
    """Random Forest classifier implemented from scratch."""
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, 
                 max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X, y):
        """Build a forest of decision trees."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        
        self.trees = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """Predict class labels using majority voting."""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            final_predictions.append(np.bincount(votes.astype(int)).argmax())
        return np.array(final_predictions)


# ============================================================================
# K-Fold Cross-Validation (From Scratch)
# ============================================================================

def kfold_cross_validation(X, y, model_class, model_params, n_splits=10):
    """Perform K-fold cross-validation from scratch."""
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n_samples // n_splits
    scores = []
    
    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_splits - 1 else n_samples
        
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        accuracy = np.mean(predictions == y_test)
        scores.append(accuracy)
    
    return np.array(scores)


# ============================================================================
# Main Execution
# ============================================================================

def run_algorithms(X, y, dataset_name):
    """Run both algorithms on a dataset with 10-fold cross-validation."""
    print(f"\nDataset: {dataset_name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print("-"*50)
    
    # Scale data for SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SVM with RBF kernel
    print("Running SVM with RBF kernel...")
    svm_params = {'C': 1.0, 'gamma': 'scale', 'max_iter': 1000, 'tol': 1e-3}
    svm_scores = kfold_cross_validation(X_scaled, y, SVMFromScratch, svm_params, n_splits=10)
    print(f"SVM (RBF):        {svm_scores.mean():.4f} ± {svm_scores.std():.4f}")
    
    # Random Forest
    print("Running Random Forest...")
    rf_params = {'n_estimators': 100, 'max_depth': 10, 'max_features': 'sqrt', 'random_state': 42}
    rf_scores = kfold_cross_validation(X, y, RandomForestFromScratch, rf_params, n_splits=10)
    print(f"Random Forest:    {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    
    return svm_scores.mean(), rf_scores.mean()


if __name__ == "__main__":
    print("="*50)
    print("Madison Rose Lucas (MRL58)")
    print("CS 634 Final Project - Option 1")
    print("Algorithms Implemented From Scratch")
    print("="*50)
    
    results = {}
    
    # Dataset 1: Iris
    print("\n" + "="*50)
    iris = load_iris()
    svm1, rf1 = run_algorithms(iris.data, iris.target, "Iris")
    results['Iris'] = {'SVM': svm1, 'RF': rf1}
    
    # Dataset 2: Wine  
    print("\n" + "="*50)
    wine = load_wine()
    svm2, rf2 = run_algorithms(wine.data, wine.target, "Wine")
    results['Wine'] = {'SVM': svm2, 'RF': rf2}
    
    # Dataset 3: Breast Cancer
    print("\n" + "="*50)
    cancer = load_breast_cancer()
    svm3, rf3 = run_algorithms(cancer.data, cancer.target, "Breast Cancer")
    results['Breast Cancer'] = {'SVM': svm3, 'RF': rf3}
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for dataset, scores in results.items():
        print(f"{dataset:15s}: SVM={scores['SVM']:.4f}, RF={scores['RF']:.4f}")
    print("="*50)
