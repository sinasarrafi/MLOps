# Demonstrate effect of different C values on Logistic Regression
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Set random seed for reproducibility
np.random.seed(42)

# Fetch dataset from UCI ML Repository
heart_disease = fetch_ucirepo(id=45)

# Extract features and targets
X = heart_disease.data.features
y = heart_disease.data.targets.squeeze()  # Convert to Series

# Remove rows with NaNs
X_clean = X.dropna()
y_clean = y[X_clean.index]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

# Scale features for better convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 60)
print("Comparing different C values (Logistic Regression)")
print("=" * 60)

C_values = [0.01, 0.1, 1.0, 10, 100]

for C in C_values:
    lr_model_test = LogisticRegression(max_iter=1000, C=C)
    lr_model_test.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, lr_model_test.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, lr_model_test.predict(X_test_scaled))
    
    print(f"\nC={C:6.2f}:")
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy:  {test_acc:.3f}")
    print(f"  Overfit Gap:    {(train_acc - test_acc):.3f}")

print("\n" + "=" * 60)
print("Comparing different max_depth values (Decision Tree)")
print("=" * 60)

depth_values = [2, 5, 10, 20, None]

for depth in depth_values:
    dt_model_test = DecisionTreeClassifier(random_state=42, max_depth=depth)
    dt_model_test.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, dt_model_test.predict(X_train))
    test_acc = accuracy_score(y_test, dt_model_test.predict(X_test))
    
    depth_str = str(depth) if depth else "None"
    print(f"\nmax_depth={depth_str:4s}:")
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy:  {test_acc:.3f}")
    print(f"  Overfit Gap:    {(train_acc - test_acc):.3f}")

print("\n" + "=" * 60)