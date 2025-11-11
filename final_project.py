"""
CS 634 Data Mining - Final Term Project
Option 1: Supervised Data Mining (Classification)

Student Name: Madison Rose Lucas
NJIT UCID: MRL58
Email: MRL58@njit.edu

Algorithm 1: LIBSVM with RBF Kernel (Category 1: Support Vector Machines)
Algorithm 2: Random Forest (Category 2: Random Forests)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

def run_algorithms(X, y, dataset_name):
    print(f"\nDataset: {dataset_name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print("-"*50)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # LIBSVM
    svm = SVC(kernel='rbf', random_state=42)
    svm_scores = cross_val_score(svm, X_scaled, y, cv=kfold)
    print(f"LIBSVM (RBF):     {svm_scores.mean():.4f} ± {svm_scores.std():.4f}")
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=kfold)
    print(f"Random Forest:    {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    
    return svm_scores.mean(), rf_scores.mean()

# Main execution
print("="*50)
print("Madison Rose Lucas (MRL58)")
print("CS 634 Final Project - Option 1")
print("="*50)

results = {}

# Dataset 1: Iris
iris = load_iris()
svm1, rf1 = run_algorithms(iris.data, iris.target, "Iris")
results['Iris'] = {'SVM': svm1, 'RF': rf1}

# Dataset 2: Wine  
wine = load_wine()
svm2, rf2 = run_algorithms(wine.data, wine.target, "Wine")
results['Wine'] = {'SVM': svm2, 'RF': rf2}

# Dataset 3: Breast Cancer
cancer = load_breast_cancer()
svm3, rf3 = run_algorithms(cancer.data, cancer.target, "Breast Cancer")
results['Breast Cancer'] = {'SVM': svm3, 'RF': rf3}

# Summary
print("\n" + "="*50)
print("Summary")
print("="*50)
for dataset, scores in results.items():
    print(f"{dataset}: SVM={scores['SVM']:.4f}, RF={scores['RF']:.4f}")
