from sklearn.metrics import accuracy_score

def evaluate(model, scaler, X_test, y_test):
    """Evaluate on the test set and print accuracy."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.3f}")
    # Optionally: log metrics, plots, and artifacts