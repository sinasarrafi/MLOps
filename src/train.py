from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train(X_train, y_train, max_iter: int = 1000, random_state: int = 42):
    """Scale features and train Logistic Regression."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model, scaler